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

import os
import pickle
from typing import Optional, Dict, Iterable, Callable

import jax
import jax.numpy as jn
import numpy as np
from absl import flags
from tqdm import tqdm, trange

import objax
from examples.classify.semi_supervised.img.libml.data.core import DataSet
from objax.util import EasyDict

FLAGS = flags.FLAGS


class TrainLoopFSL(objax.Module):
    model: objax.Module
    eval_op: Callable
    train_op: Callable

    def __init__(self, nclass: int, **kwargs):
        self.params = EasyDict(kwargs)
        self.nclass = nclass

    def serialize_model(self):  # Overload it in your model if you need something different.
        return pickle.dumps(self.model)

    def print(self):
        print(self.model.vars())
        print('Byte size %d\n' % len(self.serialize_model()))
        print('Parameters'.center(79, '-'))
        for kv in sorted(self.params.items()):
            print('%-32s %s' % kv)

    def train_step(self, summary: objax.jaxboard.Summary, data: dict, step: np.ndarray):
        kv = self.train_op(step, data['image'], data['label'])
        for k, v in kv.items():
            if jn.isnan(v):
                raise ValueError('NaN', k)
            summary.scalar(k, float(v))

    def eval(self, summary: objax.jaxboard.Summary, epoch: int, test: Dict[str, Iterable],
             valid: Optional[Iterable] = None):
        def get_accuracy(dataset: DataSet):
            accuracy, total, batch = 0, 0, None
            for data in tqdm(dataset, leave=False, desc='Evaluating'):
                x, y = data['image'].numpy(), data['label'].numpy()
                total += x.shape[0]
                batch = batch or x.shape[0]
                if x.shape[0] != batch:
                    # Pad the last batch if it's smaller than expected (must divide properly on GPUs).
                    x = np.concatenate([x] + [x[-1:]] * (batch - x.shape[0]))
                p = self.eval_op(x)[:y.shape[0]]
                accuracy += (np.argmax(p, axis=1) == data['label'].numpy()).sum()
            return accuracy / total if total else 0

        valid_accuracy = 0 if valid is None else get_accuracy(valid)
        summary.scalar('accuracy/valid', 100 * valid_accuracy)
        test_accuracy = {key: get_accuracy(value) for key, value in test.items()}
        to_print = []
        for key, value in sorted(test_accuracy.items()):
            summary.scalar('accuracy/%s' % key, 100 * value)
            to_print.append('Acccuracy/%s %.2f' % (key, summary['accuracy/%s' % key]()))
        print('Epoch %-4d  Loss %.2f  %s (Valid %.2f)' % (epoch + 1, summary['losses/xe'](), ' '.join(to_print),
                                                          summary['accuracy/valid']()))

    def train(self, train_kimg: int, report_kimg: int, train: Iterable, valid: Iterable,
              test: Dict[str, Iterable], logdir: str, keep_ckpts: int, verbose: bool = True):
        if verbose:
            self.print()
            print()
            print('Training config'.center(79, '-'))
            print('%-20s %s' % ('Test sets:', sorted(test.keys())))
            print('%-20s %s' % ('Work directory:', logdir))
            print()
        model_path = os.path.join(logdir, 'model/latest.pickle')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        ckpt = objax.io.Checkpoint(logdir=logdir, keep_ckpts=keep_ckpts)
        start_epoch = ckpt.restore(self.vars())[0]

        train_iter = iter(train)
        step_array = np.zeros(jax.device_count(), 'uint32')  # for multi-GPU
        with objax.jaxboard.SummaryWriter(os.path.join(logdir, 'tb')) as tensorboard:
            for epoch in range(start_epoch, train_kimg // report_kimg):
                summary = objax.jaxboard.Summary()
                loop = trange(0, report_kimg << 10, self.params.batch,
                              leave=False, unit='img', unit_scale=self.params.batch,
                              desc='Epoch %d/%d' % (1 + epoch, train_kimg // report_kimg))
                with self.vars().replicate():
                    for step in loop:
                        step_array[:] = step + (epoch * (report_kimg << 10))
                        self.train_step(summary, next(train_iter), step=step_array)

                    self.eval(summary, epoch, test, valid)

                tensorboard.write(summary, step=(epoch + 1) * report_kimg * 1024)
                ckpt.save(self.vars(), epoch + 1)
                with open(model_path, 'wb') as f:
                    f.write(self.serialize_model())


class TrainLoopSSL(TrainLoopFSL):
    def train_step(self, summary: objax.jaxboard.Summary, data: dict, step: np.ndarray):
        kv = self.train_op(step, data['x'], data['u'], data['label'])
        for k, v in kv.items():
            if jn.isnan(v):
                raise ValueError('NaN', k)
            summary.scalar(k, float(v))
