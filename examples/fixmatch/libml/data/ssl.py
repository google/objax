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
from typing import Callable, List

from absl import flags

from examples.fixmatch.libml.data import core

FLAGS = flags.FLAGS


class DataSetsUnlabeled:
    def __init__(self, name: str, train: core.DataSet):
        self.name = name
        self.train = train

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
    def creator(cls, name: str, train_files: List[str], parse_fn: Callable = core.record_parse,
                height: int = 32, width: int = 32, colors: int = 3, cache: bool = False):
        train_files = [os.path.join(core.DATA_DIR, x) for x in train_files]

        def create():
            image_shape = height, width, colors
            kw = dict(parse_fn=parse_fn)
            train = core.DataSet.from_files(train_files, image_shape, **kw)
            if cache:
                train = train.cache()
            return cls(name, train)

        return name, create


def create_datasets():
    d = {}
    d.update([DataSetsUnlabeled.creator('mnist', ['mnist-train.tfrecord'], cache=True,
                                        parse_fn=core.record_parse_mnist)])
    d.update([DataSetsUnlabeled.creator('cifar10', ['cifar10-train.tfrecord'], cache=True)])
    d.update([DataSetsUnlabeled.creator('cifar100', ['cifar100-train.tfrecord'], cache=True)])
    d.update([DataSetsUnlabeled.creator('svhn', ['SSL/svhn-unlabel.tfrecord'])])
    d.update([DataSetsUnlabeled.creator('svhnx', ['SSL/svhnx-unlabel.tfrecord'])])
    d.update([DataSetsUnlabeled.creator('stl10', ['SSL/stl10-unlabel.tfrecord'], height=96, width=96)])
    return d


DATASETS_UNLABELED = create_datasets
