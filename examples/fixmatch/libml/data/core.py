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
from typing import Callable, Optional, Tuple, List, Iterable

import numpy as np
import tensorflow as tf
from absl import flags

DATA_DIR = os.path.join(os.environ['ML_DATA'], os.environ['PROJECT'])

flags.DEFINE_integer('para_parse', 1, 'Parallel parsing.')
flags.DEFINE_integer('para_augment', 4, 'Parallel augmentation.')
flags.DEFINE_integer('shuffle', 8192, 'Size of dataset shuffling.')

FLAGS = flags.FLAGS


def record_parse(index: int, record: str, image_shape: Tuple[int, int, int]):
    features = tf.io.parse_single_example(record,
                                          features={'image': tf.io.FixedLenFeature([], tf.string),
                                                    'label': tf.io.FixedLenFeature([], tf.int64)})
    image = tf.image.decode_image(features['image'])
    image.set_shape(image_shape)
    image = tf.cast(image, tf.float32) * (2.0 / 255) - 1.0
    return dict(index=index, image=image, label=features['label'])


def record_parse_mnist(index: int, record: str, image_shape: Tuple[int, int, int]):
    del image_shape
    features = tf.io.parse_single_example(record,
                                          features={'image': tf.io.FixedLenFeature([], tf.string),
                                                    'label': tf.io.FixedLenFeature([], tf.int64)})
    image = tf.image.decode_image(features['image'])
    image = tf.pad(image, [(2, 2), (2, 2), (0, 0)])
    image.set_shape((32, 32, 3))
    image = tf.cast(image, tf.float32) * (2.0 / 255) - 1.0
    return dict(index=index, image=image, label=features['label'])


class DataSet:
    """Wrapper for tf.data.Dataset to permit extensions."""

    def __init__(self, data: tf.data.Dataset,
                 image_shape: Tuple[int, int, int],
                 parse_fn: Optional[Callable] = record_parse):
        self.data = data
        self.parse_fn = parse_fn
        self.image_shape = image_shape

    @classmethod
    def from_arrays(cls, images: np.ndarray, labels: np.ndarray):
        return cls(tf.data.Dataset.from_tensor_slices(dict(image=images, label=labels,
                                                           index=np.arange(images.shape[0], dtype=np.int64))),
                   images.shape[1:], parse_fn=None)

    @classmethod
    def from_files(cls, filenames: List[str],
                   image_shape: Tuple[int, int, int],
                   parse_fn: Optional[Callable] = record_parse):
        filenames_in = filenames
        filenames = sorted(sum([tf.io.gfile.glob(x) for x in filenames], []))
        if not filenames:
            raise ValueError('Empty dataset, files not found:', filenames_in)
        return cls(tf.data.TFRecordDataset(filenames).enumerate(), image_shape, parse_fn=parse_fn)

    @classmethod
    def from_tfds(cls, dataset: tf.data.Dataset, image_shape: Tuple[int, int, int]):
        d = dataset.enumerate()
        d = d.map(lambda index, x: dict(image=tf.cast(x['image'], tf.float32) / 127.5 - 1, label=x['label'], index=x))
        return cls(d, image_shape, parse_fn=None)

    def __iter__(self):
        return iter(self.data)

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]

        def call_and_update(*args, **kwargs):
            v = getattr(self.__dict__['data'], item)(*args, **kwargs)
            if isinstance(v, tf.data.Dataset):
                return self.__class__(v, self.image_shape, parse_fn=self.parse_fn)
            return v

        return call_and_update

    def dmap(self, f: Callable):
        return self.map(lambda x: f(**x))

    def nchw(self):
        return self.dmap(lambda image, **kw: dict(image=tf.transpose(image, [0, 3, 1, 2]), **kw))

    def one_hot(self, nclass):
        return self.dmap(lambda label, **kw: dict(label=tf.one_hot(label, nclass), **kw))

    def parse(self, para=None):
        if not self.parse_fn:
            return self
        para = para or FLAGS.para_parse
        if self.image_shape:
            return self.map(lambda index, record: self.parse_fn(index, record, self.image_shape), para)
        return self.map(lambda index, record: self.parse_fn(index, record), para)

    def __len__(self):
        count = 0
        for _ in self.data:
            count += 1
        return count


class Numpyfier:
    def __init__(self, dataset: Iterable):
        self.dataset = dataset

    def __iter__(self):
        for d in self.dataset:
            yield {k: v.numpy() for k, v in d.items()}


def tiny_parse(index: int, record: str, image_shape: Tuple[int, int, int]):
    del image_shape
    features = tf.io.parse_single_example(record,
                                          features={'image': tf.io.FixedLenFeature([], tf.string),
                                                    'label': tf.io.FixedLenFeature([], tf.int64)})
    image = tf.reshape(tf.io.decode_raw(features['image'], tf.uint8), [3, 32, 32])
    image = tf.transpose(image, [2, 1, 0])
    image = tf.cast(image, tf.float32) * (2.0 / 255) - 1.0
    return dict(index=index, image=image, label=features['label'])


def record_parse_stl10_32(index: int, record: str, image_shape: Tuple[int, int, int]):
    features = tf.io.parse_single_example(record,
                                          features={'image': tf.io.FixedLenFeature([], tf.string),
                                                    'label': tf.io.FixedLenFeature([], tf.int64)})
    image = tf.image.decode_image(features['image'])
    image = tf.cast(image, tf.float32) * (2.0 / 255) - 1.0
    image = tf.nn.avg_pool2d([image], 3, 3, 'VALID')[0]
    image.set_shape(image_shape)
    return dict(index=index, image=image, label=features['label'])
