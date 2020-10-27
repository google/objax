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
import inspect
import multiprocessing
from typing import Tuple, Callable, Union, List, Type, Optional

import numpy as np
import tensorflow as tf
from absl import flags

import objax
from examples.fixmatch.libml.augment import ctaugment
from examples.fixmatch.libml.augment.core import get_tf_augment
from examples.fixmatch.libml.data import core

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'augment', 'CTA(sm,sm,sm)',
    'Dataset augmentation method:\n'
    '  Augmentation primitives:\n'
    '    x   = identity\n'
    '    m   = mirror\n'
    '    s   = shift\n'
    '    sc  = shift+cutout\n'
    '    sm  = shift+mirror\n'
    '    smc = shift+mirror+cutout\n'
    'Augmentations:\n'
    '  (primitive,primitive,primitive) = Default augmentation (primitives for labeled, 1st unlabeled, 2nd unlabeled)\n'
    '  CTA(primitive,primitive,primitive,depth=int,th=float,decay=float) = CTAugment\n')


class AugmentPool:
    NAME = ''

    def __init__(self, labeled: core.DataSet, unlabeled: core.DataSet, nclass: int, batch: int, uratio: int,
                 tfops: Tuple[Callable, Callable, Callable], predict: Callable):
        x = labeled.map(tfops[0], FLAGS.para_augment).batch(batch).nchw().one_hot(nclass)
        u = unlabeled.map(lambda d: dict(image=tf.stack([tfops[1](d)['image'], tfops[2](d)['image']]),
                                         index=d['index'], label=d['label']),
                          FLAGS.para_augment)
        u = u.batch(batch * uratio).map(lambda d: dict(image=tf.transpose(d['image'], [0, 1, 4, 2, 3]),
                                                       index=d['index'], label=d['label']))
        train = x.zip((x.data, u.data)).map(lambda x, u: dict(xindex=x['index'], x=x['image'], label=x['label'],
                                                              uindex=u['index'], u=u['image'], secret_label=u['label']))
        self.train = core.Numpyfier(train.prefetch(16))
        self.predict = predict

    def __iter__(self):
        return iter(self.train)


class AugmentPoolCTA(AugmentPool):
    NAME = 'CTA'

    def __init__(self, labeled: core.DataSet, unlabeled: core.DataSet, nclass: int, batch: int, uratio: int,
                 tfops: Tuple[Callable, Callable, Callable], predict: Callable,
                 depth: Union[int, str] = 2,
                 th: Union[float, str] = 0.8,
                 decay: Union[float, str] = 0.99):
        super().__init__(labeled, unlabeled, nclass, batch, uratio, tfops, predict)
        self.pool = multiprocessing.Pool(FLAGS.para_augment)
        self.cta = ctaugment.CTAugment(int(depth), float(th), float(decay))
        self.queue = []

    @staticmethod
    def numpy_apply_policies(x, u, cta: ctaugment.CTAugment):
        x = objax.util.image.nhwc(x)
        u = objax.util.image.nhwc(u)
        nchw = objax.util.image.nchw
        policy_list = [cta.policy(probe=True) for _ in range(x.shape[0])]
        cutout_policy = lambda probe: cta.policy(probe=probe) + [ctaugment.OP('cutout', (1,))]
        u_strong = np.stack([ctaugment.apply(u[i, 1], cutout_policy(False)) for i in range(u.shape[0])])
        return dict(policy=policy_list,
                    probe=nchw(np.stack([ctaugment.apply(x[i], policy) for i, policy in enumerate(policy_list)])),
                    u=nchw(np.stack([u[:, 0], u_strong], axis=1)))

    def __iter__(self):
        for data in self.train:
            self.queue.append((dict(xindex=data['xindex'], uindex=data['uindex'], x=data['x'],
                                    label=data['label'], secret_label=data['secret_label']),
                               self.pool.apply_async(self.numpy_apply_policies, (data['x'], data['u'], self.cta))))
            if len(self.queue) > 2 * FLAGS.para_augment:
                d, pd = self.queue.pop(0)
                d.update(pd.get())
                probe = self.predict(d['probe'])
                w1 = 1 - 0.5 * np.abs(probe - d['label']).sum(1)
                for p in range(w1.shape[0]):
                    self.cta.update_rates(d['policy'][p], w1[p])
                yield d


def get_augment(train: core.DataSet, extra: Optional[List[Type[AugmentPool]]] = None):
    pool, a = FLAGS.augment.split('(')
    a = a[:-1].split(',')
    a, kwargs = a[:3], {k: v for k, v in (x.split('=') for x in a[3:])}
    kwargs['tfops'] = tuple(get_tf_augment(ai, size=train.image_shape[0]) for ai in a)
    for x in list(globals().values()) + (extra or []):
        if inspect.isclass(x) and issubclass(x, AugmentPool):
            if x.NAME == pool:
                pool = x
    return functools.partial(pool, **kwargs)
