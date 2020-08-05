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

import itertools
import os
from typing import List, Callable, Dict

import numpy as np
from absl import flags

from examples.classify.semi_supervised.img.libml.data import core

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'cifar10-5000', 'Data to train on.')


class DataSetsLabeled:
    def __init__(self, name: str, train: core.DataSet, test: Dict[str, core.DataSet], valid: core.DataSet,
                 nclass: int = 10):
        self.name = name
        self.train = train
        self.test = test
        self.valid = valid
        self.nclass = nclass

    @property
    def colors(self):
        return self.train.image_shape[2]

    @property
    def height(self):
        return self.train.image_shape[0]

    @property
    def width(self):
        return self.train.image_shape[1]

    @classmethod
    def creator(cls, name: str, train_files: List[str], test_files: Dict[str, List[str]], valid: int,
                parse_fn: Callable = core.record_parse,
                nclass: int = 10, height: int = 32, width: int = 32, colors: int = 3, cache: bool = False):
        train_files = [os.path.join(core.DATA_DIR, x) for x in train_files]
        test_files = {key: [os.path.join(core.DATA_DIR, x) for x in value] for key, value in test_files.items()}

        def create():
            image_shape = height, width, colors
            kw = dict(parse_fn=parse_fn)
            datasets = dict(train=core.DataSet.from_files(train_files, image_shape, **kw).skip(valid),
                            valid=core.DataSet.from_files(train_files, image_shape, **kw).take(valid),
                            test={key: core.DataSet.from_files(value, image_shape, **kw) for key, value in
                                  test_files.items()})
            if cache:
                cached_datasets = {}
                for key, value in datasets.items():
                    if isinstance(value, dict):
                        cached_datasets[key] = {k: v.cache() for k, v in value.items()}
                    else:
                        cached_datasets[key] = value.cache()
                datasets = cached_datasets
            return cls(name + '-' + str(valid), nclass=nclass, **datasets)

        return name + '-' + str(valid), create


def create_datasets(samples_per_class=(1, 2, 3, 4, 5, 10, 25, 100, 400)):
    samples_per_class = np.array(samples_per_class, np.uint32)
    d = {}
    d.update([DataSetsLabeled.creator('mnist', ['mnist-train.tfrecord'], {'mnist': ['mnist-test.tfrecord']}, valid,
                                      cache=True, parse_fn=core.record_parse_mnist) for valid in [0, 5000]])
    d.update(
        [DataSetsLabeled.creator('cifar10', ['cifar10-train.tfrecord'], {'cifar10': ['cifar10-test.tfrecord']}, valid,
                                 cache=True) for valid in [0, 5000]])
    d.update(
        [DataSetsLabeled.creator('cifar100', ['cifar100-train.tfrecord'], {'cifar100': ['cifar100-test.tfrecord']},
                                 valid,
                                 nclass=100, cache=True) for valid in [0, 5000]])
    d.update([DataSetsLabeled.creator('svhn', ['svhn-train.tfrecord'], {'svhn': ['svhn-test.tfrecord']},
                                      valid) for valid in [0, 5000]])
    d.update([DataSetsLabeled.creator('svhnx', ['svhn-train.tfrecord', 'svhn-extra.tfrecord'],
                                      {'svhn': ['svhn-test.tfrecord']}, valid) for valid in [0, 5000]])
    d.update([DataSetsLabeled.creator('cifar10.%d@%d' % (seed, sz), ['SSL/cifar10.%d@%d-label.tfrecord' % (seed, sz)],
                                      {'cifar10': ['cifar10-test.tfrecord']}, valid, cache=True)
              for valid, seed, sz in itertools.product([0, 5000], range(6), 10 * samples_per_class)])
    d.update([DataSetsLabeled.creator('cifar100.%d@%d' % (seed, sz), ['SSL/cifar100.%d@%d-label.tfrecord' % (seed, sz)],
                                      {'cifar100': ['cifar100-test.tfrecord']}, valid, nclass=100)
              for valid, seed, sz in itertools.product([0, 5000], range(6), 100 * samples_per_class)])
    d.update([DataSetsLabeled.creator('svhn.%d@%d' % (seed, sz), ['SSL/svhn.%d@%d-label.tfrecord' % (seed, sz)],
                                      {'svhn': ['svhn-test.tfrecord']}, valid)
              for valid, seed, sz in itertools.product([0, 5000], range(6), 10 * samples_per_class)])
    d.update([DataSetsLabeled.creator('svhnx.%d@%d' % (seed, sz), ['SSL/svhnx.%d@%d-label.tfrecord' % (seed, sz)],
                                      {'svhn': ['svhn-test.tfrecord']}, valid)
              for valid, seed, sz in itertools.product([0, 5000], range(6), 10 * samples_per_class)])
    d.update([DataSetsLabeled.creator('stl10.%d@%d' % (seed, sz), ['SSL/stl10.%d@%d-label.tfrecord' % (seed, sz)],
                                      {'stl10': ['stl10-test.tfrecord']}, valid, height=96, width=96)
              for valid, seed, sz in itertools.product([0, 5000], range(6), 10 * samples_per_class)])
    return d


DATASETS_LABELED = create_datasets
