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

"""Augmentations for images.
"""

import tensorflow as tf


def cutout(x, w):
    offsets = tf.random.uniform([2], 0, 1)
    s = tf.shape(x)
    y0 = tf.cast(tf.round(offsets[0] * (tf.cast(s[0], tf.float32) - w)), tf.int32)
    x0 = tf.cast(tf.round(offsets[1] * (tf.cast(s[1], tf.float32) - w)), tf.int32)
    hr, wr = tf.range(s[0])[:, None, None], tf.range(s[1])[None, :, None]
    mask = 1-tf.cast((hr >= y0) & (hr < y0 + w) & (wr >= x0) & (wr < x0 + w), tf.float32)
    return mask * x


def mirror(x):
    return tf.image.random_flip_left_right(x)


def shift(x, w):
    y = tf.pad(x, [[w] * 2, [w] * 2, [0] * 2], mode='REFLECT')
    return tf.image.random_crop(y, tf.shape(x))


def noise(x, std):
    return x + std * tf.random.normal(tf.shape(x), dtype=x.dtype)


def get_tf_augment(augment, size=32):
    aug = dict(
        x=lambda **kw: kw,
        s=lambda image, **kw: dict(image=shift(image, size >> 3), **kw),
        sc=lambda image, **kw: dict(image=cutout(shift(image, size >> 3), size >> 1), **kw),
        sm=lambda image, **kw: dict(image=mirror(shift(image, size >> 3)), **kw),
        smc=lambda image, **kw: dict(image=cutout(mirror(shift(image, size >> 3)), size >> 1), **kw))
    return lambda x: aug[augment](**x)
