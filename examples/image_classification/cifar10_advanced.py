# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import os
from typing import Callable

import jax
import jax.numpy as jn
import numpy as np
import tensorflow as tf  # For data augmentation.
import tensorflow_datasets as tfds
from absl import app, flags
from tqdm import tqdm, trange

import objax
from examples.classify.img.tfdata.data import DataSet
from objax.jaxboard import SummaryWriter, Summary
from objax.util import EasyDict
from objax.zoo import convnet, wide_resnet

FLAGS = flags.FLAGS


def augment(x, shift: int):
    y = tf.image.random_flip_left_right(x['image'])
    y = tf.pad(y, [[shift] * 2, [shift] * 2, [0] * 2], mode='REFLECT')
    return dict(image=tf.image.random_crop(y, tf.shape(x['image'])), label=x['label'])


# We make our own TrainLoop to be reusable
class TrainLoop(objax.Module):
    predict: Callable
    train_op: Callable

    def __init__(self, nclass: int, **kwargs):
        self.nclass = nclass
        self.params = EasyDict(kwargs)

    def train_step(self, summary: Summary, data: dict, progress: np.ndarray):
        kv = self.train_op(progress, data['image'].numpy(), data['label'].numpy())
        for k, v in kv.items():
            if jn.isnan(v):
                raise ValueError('NaN, try reducing learning rate', k)
            summary.scalar(k, float(v))

    def train(self, num_train_epochs: int, train_size: int, train: DataSet, test: DataSet, logdir: str):
        checkpoint = objax.io.Checkpoint(logdir, keep_ckpts=5, makedir=True)
        start_epoch, last_ckpt = checkpoint.restore(self.vars())
        train_iter = iter(train)
        progress = np.zeros(jax.local_device_count(), 'f')  # for multi-GPU

        with SummaryWriter(os.path.join(logdir, 'tb')) as tensorboard:
            for epoch in range(start_epoch, num_train_epochs):
                with self.vars().replicate():
                    # Train
                    summary = Summary()
                    loop = trange(0, train_size, self.params.batch,
                                  leave=False, unit='img', unit_scale=self.params.batch,
                                  desc='Epoch %d/%d' % (1 + epoch, num_train_epochs))
                    for step in loop:
                        progress[:] = (step + (epoch * train_size)) / (num_train_epochs * train_size)
                        self.train_step(summary, next(train_iter), progress)

                    # Eval
                    accuracy, total = 0, 0
                    for data in tqdm(test, leave=False, desc='Evaluating'):
                        total += data['image'].shape[0]
                        preds = np.argmax(self.predict(data['image'].numpy()), axis=1)
                        accuracy += (preds == data['label'].numpy()).sum()
                    accuracy /= total
                    summary.scalar('eval/accuracy', 100 * accuracy)
                    print('Epoch %04d  Loss %.2f  Accuracy %.2f' % (epoch + 1, summary['losses/xe'](),
                                                                    summary['eval/accuracy']()))
                    tensorboard.write(summary, step=(epoch + 1) * train_size)

                checkpoint.save(self.vars(), epoch + 1)


# We inherit from the training loop and define predict and train_op.
class TrainModule(TrainLoop):
    def __init__(self, model: Callable, nclass: int, **kwargs):
        super().__init__(nclass, **kwargs)
        self.model = model(3, nclass)
        model_vars = self.model.vars()
        self.opt = objax.optimizer.Momentum(model_vars)
        self.ema = objax.optimizer.ExponentialMovingAverage(model_vars, momentum=0.999, debias=True)
        print(model_vars)

        def loss(x, label):
            logit = self.model(x, training=True)
            loss_wd = 0.5 * sum((v.value ** 2).sum() for k, v in model_vars.items() if k.endswith('.w'))
            loss_xe = objax.functional.loss.cross_entropy_logits(logit, label).mean()
            return loss_xe + loss_wd * self.params.weight_decay, {'losses/xe': loss_xe, 'losses/wd': loss_wd}

        gv = objax.GradValues(loss, model_vars)

        def train_op(progress, x, y):
            g, v = gv(x, y)
            lr = self.params.lr * jn.cos(progress * (7 * jn.pi) / (2 * 8))
            self.opt(lr, objax.functional.parallel.pmean(g))
            self.ema()
            return objax.functional.parallel.pmean({'monitors/lr': lr, **v[1]})

        def predict_op(x):
            return objax.functional.softmax(self.model(x, training=False))

        self.predict = objax.Parallel(self.ema.replace_vars(predict_op), model_vars + self.ema.vars())
        self.train_op = objax.Parallel(train_op, self.vars(), reduce=lambda x: x[0])


def network(arch: str):
    if arch == 'cnn32-3-max':
        return functools.partial(convnet.ConvNet, scales=3, filters=32, filters_max=1024,
                                 pooling=objax.functional.max_pool_2d)
    elif arch == 'cnn32-3-mean':
        return functools.partial(convnet.ConvNet, scales=3, filters=32, filters_max=1024,
                                 pooling=objax.functional.average_pool_2d)
    elif arch == 'cnn64-3-max':
        return functools.partial(convnet.ConvNet, scales=3, filters=64, filters_max=1024,
                                 pooling=objax.functional.max_pool_2d)
    elif arch == 'cnn64-3-mean':
        return functools.partial(convnet.ConvNet, scales=3, filters=64, filters_max=1024,
                                 pooling=objax.functional.average_pool_2d)
    elif arch == 'wrn28-1':
        return functools.partial(wide_resnet.WideResNet, depth=28, width=1)
    elif arch == 'wrn28-2':
        return functools.partial(wide_resnet.WideResNet, depth=28, width=2)
    raise ValueError('Architecture not recognized', arch)


def main(argv):
    del argv
    # In this example we use tensorflow_datasets for loading cifar10, but you can use any dataset library you like.
    tf.config.experimental.set_visible_devices([], "GPU")
    DATA_DIR = os.path.join(os.environ['HOME'], 'TFDS')
    data, info = tfds.load(name='cifar10', split='train', data_dir=DATA_DIR, with_info=True)
    train_size = info.splits['train'].num_examples
    image_shape = info.features['image'].shape
    nclass = info.features['label'].num_classes
    train = DataSet.from_tfds(data, image_shape, augment_fn=lambda x: augment(x, 4))
    test = DataSet.from_tfds(tfds.load(name='cifar10', split='test', data_dir=DATA_DIR), image_shape)
    train = train.cache().shuffle(8192).repeat().parse().augment().batch(FLAGS.batch)
    train = train.nchw().one_hot(nclass).prefetch(16)
    test = test.cache().parse().batch(FLAGS.batch).nchw().prefetch(16)
    del data, info

    # Define the network and train_it
    loop = TrainModule(network(FLAGS.arch), nclass=nclass,
                       arch=FLAGS.arch,
                       lr=FLAGS.lr,
                       batch=FLAGS.batch,
                       epochs=FLAGS.epochs,
                       weight_decay=FLAGS.weight_decay)
    logdir = '%s/%s' % (loop.__class__.__name__, '_'.join(sorted('%s_%s' % k for k in loop.params.items())))
    logdir = os.path.join(FLAGS.logdir, logdir)
    print(f'Saving to {logdir}')
    print(f'Visualize results with:\n    tensorboard --logdir {FLAGS.logdir}')
    loop.train(FLAGS.epochs, train_size, train, test, logdir)


if __name__ == '__main__':
    flags.DEFINE_enum('arch', 'wrn28-2', ['cnn32-3-max', 'cnn32-3-mean',
                                          'cnn64-3-max', 'cnn64-3-mean',
                                          'wrn28-1', 'wrn28-2'],
                      'Model architecture.')
    flags.DEFINE_float('lr', 0.1, 'Learning rate.')
    flags.DEFINE_float('weight_decay', 0.0005, 'Weight decay ratio.')
    flags.DEFINE_integer('batch', 256, 'Batch size')
    flags.DEFINE_integer('epochs', 1000, 'Training duration in number of epochs.')
    flags.DEFINE_string('logdir', 'experiments', 'Directory where to save checkpoints and tensorboard data.')
    app.run(main)
