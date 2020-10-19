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

"""FixMatch.

https://arxiv.org/abs/2001.07685
"""

import os
from typing import Callable

import jax.numpy as jn
from absl import app, flags

import objax
from examples.classify.semi_supervised.img.libml.augment.augment import get_augment
from examples.classify.semi_supervised.img.libml.data.fsl import DATASETS_LABELED
from examples.classify.semi_supervised.img.libml.data.ssl import DATASETS_UNLABELED
from examples.classify.semi_supervised.img.libml.models import ARCHS, network
from examples.classify.semi_supervised.img.libml.train import TrainLoopSSL
from examples.classify.semi_supervised.img.libml.util import setup_tf

FLAGS = flags.FLAGS


class FixMatch(TrainLoopSSL):
    def __init__(self, nclass: int, model: Callable, **kwargs):
        super().__init__(nclass, **kwargs)
        self.model = model(nin=3, nclass=nclass, **kwargs)
        model_vars = self.model.vars()
        self.ema = objax.optimizer.ExponentialMovingAverage(model_vars.subset(objax.TrainVar),
                                                            momentum=0.999, debias=False)
        self.opt = objax.optimizer.Momentum(model_vars, momentum=0.9, nesterov=True)

        def loss_function(x, u, y):
            xu = jn.concatenate([x, u.reshape((-1,) + u.shape[2:])], axis=0)
            logit = self.model(xu, training=True)
            logit_x = logit[:x.shape[0]]
            logit_weak = logit[x.shape[0]::2]
            logit_strong = logit[x.shape[0] + 1::2]

            xe = objax.functional.loss.cross_entropy_logits(logit_x, y).mean()

            pseudo_labels = objax.functional.stop_gradient(objax.functional.softmax(logit_weak))
            pseudo_mask = (pseudo_labels.max(axis=1) >= self.params.confidence).astype(logit_weak.dtype)
            xeu = objax.functional.loss.cross_entropy_logits_sparse(logit_strong, pseudo_labels.argmax(axis=1))
            xeu = (xeu * pseudo_mask).mean()

            wd = 0.5 * sum(objax.functional.loss.l2(v.value) for k, v in model_vars.items() if k.endswith('.w'))

            loss = xe + self.params.wu * xeu + self.params.wd * wd
            return loss, {'losses/xe': xe,
                          'losses/xeu': xeu,
                          'losses/wd': wd,
                          'monitors/mask': pseudo_mask.mean()}

        gv = objax.GradValues(loss_function, model_vars)

        def train_op(step, x, u, y):
            g, v = gv(x, u, y)
            fstep = step[0] / (FLAGS.train_kimg << 10)
            lr = self.params.lr * jn.cos(fstep * (7 * jn.pi) / (2 * 8))
            self.opt(lr, objax.functional.parallel.pmean(g))
            self.ema()
            return objax.functional.parallel.pmean({'monitors/lr': lr, **v[1]})

        eval_op = self.ema.replace_vars(lambda x: objax.functional.softmax(self.model(x, training=False)))
        self.eval_op = objax.Parallel(eval_op, model_vars + self.ema.vars())
        self.train_op = objax.Parallel(train_op, self.vars(), reduce=lambda x: x[0])


def main(argv):
    del argv
    setup_tf()
    unlabeled_name = FLAGS.unlabeled or FLAGS.dataset.split('.')[0]
    unlabeled = DATASETS_UNLABELED()[unlabeled_name]()
    dataset = DATASETS_LABELED()[FLAGS.dataset]()
    testsets = [dataset.test]
    if FLAGS.test_extra:
        testsets += [DATASETS_LABELED()[x]().test for x in FLAGS.test_extra.split(',')]
    module = FixMatch(dataset.nclass, network(FLAGS.arch),
                      lr=FLAGS.lr,
                      wd=FLAGS.wd,
                      arch=FLAGS.arch,
                      batch=FLAGS.batch,
                      wu=FLAGS.wu,
                      confidence=FLAGS.confidence,
                      uratio=FLAGS.uratio,
                      filters=FLAGS.filters,
                      filters_max=FLAGS.filters_max,
                      repeat=FLAGS.repeat,
                      scales=FLAGS.scales or objax.util.ilog2(dataset.train.image_shape[0]) - 2)
    logdir = f'{FLAGS.dataset}/{unlabeled_name}/{FLAGS.augment}/{module.__class__.__name__}/%s' % (
        '_'.join(sorted('%s%s' % k for k in module.params.items())))
    logdir = os.path.join(FLAGS.logdir, logdir)
    valid = dataset.valid.parse().batch(FLAGS.batch).nchw().prefetch(16)
    test = {}
    for x in testsets:
        test.update((k, v.parse().batch(FLAGS.batch).nchw().prefetch(16)) for k, v in x.items())
    labeled = dataset.train.repeat().shuffle(FLAGS.shuffle).parse()
    unlabeled = unlabeled.train.repeat().shuffle(FLAGS.shuffle).parse()
    augment_pool = get_augment(dataset.train)
    train = augment_pool(labeled, unlabeled, dataset.nclass, FLAGS.batch, FLAGS.uratio, predict=module.eval_op)
    module.train(FLAGS.train_kimg, FLAGS.report_kimg, train, valid, test, logdir, FLAGS.keep_ckpts)


if __name__ == '__main__':
    flags.DEFINE_enum('arch', 'resnet', ARCHS, 'Model architecture.')
    flags.DEFINE_float('confidence', 0.95, 'Confidence threshold.')
    flags.DEFINE_float('lr', 0.03, 'Learning rate.')
    flags.DEFINE_float('wd', 0.0005, 'Weight decay.')
    flags.DEFINE_float('wu', 1, 'Pseudo label loss weight.')
    flags.DEFINE_integer('batch', 64, 'Batch size')
    flags.DEFINE_integer('uratio', 5, 'Unlabeled batch size ratio')
    flags.DEFINE_integer('filters', 32, 'Initial filter size of convolutions.')
    flags.DEFINE_integer('filters_max', 512, 'Max filter size for convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    flags.DEFINE_integer('scales', 0, 'Number of blocks in the classifier.')
    flags.DEFINE_integer('report_kimg', 64, 'Reporting period in kibi-images.')
    flags.DEFINE_integer('train_kimg', 1 << 16, 'Training duration in kibi-images.')
    flags.DEFINE_integer('keep_ckpts', 5, 'Number of checkpoints to keep (0 for all).')
    flags.DEFINE_string('logdir', 'experiments', 'Directory where to save checkpoints and tensorboard data.')
    flags.DEFINE_string('unlabeled', '', 'Unlabeled data to train on (leave empty to find automatically from labeled).')
    flags.DEFINE_string('test_extra', '', 'Comma-separated list of datasets on which to report test accuracy.')
    FLAGS.set_default('dataset', 'cifar10.3@250-0')
    app.run(main)
