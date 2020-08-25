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

"""Imagenet training code using ObJAX."""

import os
import time

import jax
import jax.numpy as jn
import numpy as np
from absl import app, flags, logging

import objax
from examples.classify.img.imagenet import imagenet_data
from objax.zoo.resnet_v2 import ResNet50

flags.DEFINE_string('model_dir', '', 'Model directory.')
flags.DEFINE_integer('train_device_batch_size', 64, 'Per-device training batch size.')
flags.DEFINE_integer('eval_device_batch_size', 250, 'Per-device eval batch size.')
flags.DEFINE_integer('max_eval_batches', -1, 'Maximum number of batches used for evaluation, '
                                             'zero or negative number means use all batches.')
flags.DEFINE_integer('eval_every_n_steps', 1000, 'How often to run eval.')
flags.DEFINE_float('num_train_epochs', 90, 'Number of training epochs.')
flags.DEFINE_float('base_learning_rate', 0.1, 'Base learning rate.')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight decay (L2 loss) coefficient.')
flags.DEFINE_boolean('use_sync_bn', True, 'If true then use synchronized batch normalization, '
                                          'otherwise use per-replica batch normalization.')
flags.DEFINE_string('tfds_data_dir', None, 'Optional TFDS data directory.')

FLAGS = flags.FLAGS

NUM_CLASSES = 1000


class Experiment:
    """Class with all code to run experiment."""

    def __init__(self):
        # Some constants
        total_batch_size = FLAGS.train_device_batch_size * jax.device_count()
        self.base_learning_rate = FLAGS.base_learning_rate * total_batch_size / 256
        # Create model
        bn_cls = objax.nn.SyncedBatchNorm2D if FLAGS.use_sync_bn else objax.nn.BatchNorm2D
        self.model = ResNet50(in_channels=3, num_classes=NUM_CLASSES, normalization_fn=bn_cls)
        self.model_vars = self.model.vars()
        print(self.model_vars)
        # Create parallel eval op
        self.evaluate_batch_parallel = objax.Parallel(self.evaluate_batch, self.model_vars,
                                                      reduce=lambda x: x.sum(0))
        # Create parallel training op
        self.optimizer = objax.optimizer.Momentum(self.model_vars, momentum=0.9, nesterov=True)
        self.compute_grads_loss = objax.GradValues(self.loss_fn, self.model_vars)
        self.all_vars = self.model_vars + self.optimizer.vars()
        self.train_op_parallel = objax.Parallel(
            self.train_op, self.all_vars, reduce=lambda x: x[0])
        # Summary writer
        self.summary_writer = objax.jaxboard.SummaryWriter(os.path.join(
            FLAGS.model_dir, 'tb'))

    def evaluate_batch(self, images, labels):
        logits = self.model(images, training=False)
        num_correct = jn.count_nonzero(jn.equal(jn.argmax(logits, axis=1), labels))
        return num_correct

    def run_eval(self):
        """Runs evaluation and returns top-1 accuracy."""
        test_ds = imagenet_data.load(
            imagenet_data.Split.TEST,
            is_training=False,
            batch_dims=[jax.local_device_count() * FLAGS.eval_device_batch_size],
            tfds_data_dir=FLAGS.tfds_data_dir)

        correct_pred = 0
        total_examples = 0
        for batch_index, batch in enumerate(test_ds):
            correct_pred += self.evaluate_batch_parallel(batch['images'],
                                                         batch['labels'])
            total_examples += batch['images'].shape[0]
            if ((FLAGS.max_eval_batches > 0)
                    and (batch_index + 1 >= FLAGS.max_eval_batches)):
                break

        return correct_pred / total_examples

    def loss_fn(self, images, labels):
        """Computes loss function.

        Args:
          images: tensor with images NCHW
          labels: tensors with dense labels, shape (batch_size,)

        Returns:
          Tuple (total_loss, losses_dictionary).
        """
        logits = self.model(images, training=True)
        xent_loss = objax.functional.loss.cross_entropy_logits_sparse(logits, labels).mean()
        wd_loss = FLAGS.weight_decay * 0.5 * sum((v.value ** 2).sum()
                                                 for k, v in self.model_vars.items()
                                                 if k.endswith('.w'))
        total_loss = xent_loss + wd_loss
        return total_loss, {'total_loss': total_loss,
                            'xent_loss': xent_loss,
                            'wd_loss': wd_loss}

    def learning_rate(self, epoch: float):
        """Computes learning rate for given fractional epoch."""
        # Linear warm up to base_learning_rate value for first 5 epochs.
        # Then use 1.0 * base_learning_rate until epoch 30
        # Then use 0.1 * base_learning_rate until epoch 60
        # Then use 0.01 * base_learning_rate until epoch 80
        # Then use 0.001 * base_learning_rate until until end of traning.
        lr_linear_till = 5
        boundaries = jn.array((30, 60, 80))
        values = jn.array([1., 0.1, 0.01, 0.001]) * self.base_learning_rate

        index = jn.sum(boundaries < epoch)
        lr = jn.take(values, index)
        return lr * jn.minimum(1., epoch / lr_linear_till)

    def train_op(self, images, labels, cur_epoch):
        cur_epoch = cur_epoch[0]  # because cur_epoch is array of size 1
        grads, (_, losses_dict) = self.compute_grads_loss(images, labels)
        grads = objax.functional.parallel.pmean(grads)
        losses_dict = objax.functional.parallel.pmean(losses_dict)
        learning_rate = self.learning_rate(cur_epoch)
        self.optimizer(learning_rate, grads)
        return dict(**losses_dict, learning_rate=learning_rate, epoch=cur_epoch)

    def train_and_eval(self):
        """Runs training and evaluation."""
        train_ds = imagenet_data.load(
            imagenet_data.Split.TRAIN,
            is_training=True,
            batch_dims=[jax.local_device_count() * FLAGS.train_device_batch_size],
            tfds_data_dir=FLAGS.tfds_data_dir)

        steps_per_epoch = (imagenet_data.Split.TRAIN.num_examples
                           / (FLAGS.train_device_batch_size * jax.device_count()))
        total_train_steps = int(steps_per_epoch * FLAGS.num_train_epochs)
        eval_every_n_steps = FLAGS.eval_every_n_steps

        checkpoint = objax.io.Checkpoint(FLAGS.model_dir, keep_ckpts=10)
        start_step, _ = checkpoint.restore(self.all_vars)
        cur_epoch = np.zeros([jax.local_device_count()], dtype=np.float32)
        for big_step in range(start_step, total_train_steps, eval_every_n_steps):
            print(f'Running training steps {big_step + 1} - {big_step + eval_every_n_steps}')
            with self.all_vars.replicate():
                # training
                start_time = time.time()
                for cur_step in range(big_step + 1, big_step + eval_every_n_steps + 1):
                    batch = next(train_ds)
                    cur_epoch[:] = cur_step / steps_per_epoch
                    monitors = self.train_op_parallel(
                        batch['images'], batch['labels'], cur_epoch)
                elapsed_train_time = time.time() - start_time
                # eval
                start_time = time.time()
                accuracy = self.run_eval()
                elapsed_eval_time = time.time() - start_time
            # save summary
            summary = objax.jaxboard.Summary()
            for k, v in monitors.items():
                summary.scalar(f'train/{k}', v)
            # # Uncomment following two lines to save summary with training images
            # summary.image('input/train_img',
            #               imagenet_data.normalize_image_for_view(batch['images'][0]))
            summary.scalar('test/accuracy', accuracy * 100)
            self.summary_writer.write(summary, step=cur_step)
            # save checkpoint
            checkpoint.save(self.all_vars, cur_step)
            # print info
            print('Step %d -- Epoch %.2f -- Loss %.2f  Accuracy %.2f'
                  % (cur_step, cur_step / steps_per_epoch,
                     monitors['total_loss'], accuracy * 100))
            print('    Training took %.1f seconds, eval took %.1f seconds'
                  % (elapsed_train_time, elapsed_eval_time), flush=True)


def main(argv):
    del argv
    print('JAX host: %d / %d' % (jax.host_id(), jax.host_count()))
    print('JAX devices:\n%s' % '\n'.join(str(d) for d in jax.devices()), flush=True)
    experiment = Experiment()
    experiment.train_and_eval()


if __name__ == '__main__':
    logging.set_verbosity(logging.ERROR)
    jax.config.config_with_absl()
    app.run(main)
