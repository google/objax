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

"""Imagenet dataset reader.

Code based on https://github.com/deepmind/dm-haiku/blob/master/examples/imagenet/dataset.py
"""

import enum
from typing import Optional, Sequence, Tuple

import jax
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)

IMAGE_SIZE = 224
IMAGE_PADDING_FOR_CROP = 32


class Split(enum.Enum):
    """Imagenet dataset split."""
    TRAIN = 1
    TEST = 2

    @property
    def num_examples(self):
        return {Split.TRAIN: 1281167, Split.TEST: 50000}[self]


def _to_tfds_split(split: Split) -> tfds.Split:
    """Returns the TFDS split appropriately sharded."""
    if split == Split.TRAIN:
        return tfds.Split.TRAIN
    else:
        assert split == Split.TEST
        return tfds.Split.VALIDATION


def _shard(split: Split, shard_index: int, num_shards: int) -> Tuple[int, int]:
    """Returns [start, end) for the given shard index."""
    assert shard_index < num_shards
    arange = np.arange(split.num_examples)
    shard_range = np.array_split(arange, num_shards)[shard_index]
    start, end = shard_range[0], (shard_range[-1] + 1)
    return start, end


def load(split: Split, is_training: bool, batch_dims: Sequence[int], tfds_data_dir: Optional[str] = None):
    """Loads the given split of the dataset."""
    if is_training:
        start, end = _shard(split, jax.host_id(), jax.host_count())
    else:
        start, end = _shard(split, 0, 1)
    tfds_split = tfds.core.ReadInstruction(_to_tfds_split(split),
                                           from_=start, to=end, unit='abs')
    ds = tfds.load('imagenet2012:5.*.*', split=tfds_split,
                   decoders={'image': tfds.decode.SkipDecoding()}, data_dir=tfds_data_dir)

    total_batch_size = np.prod(batch_dims)

    options = tf.data.Options()
    options.experimental_threading.private_threadpool_size = 48
    options.experimental_threading.max_intra_op_parallelism = 1
    if is_training:
        options.experimental_deterministic = False
    ds = ds.with_options(options)

    if is_training:
        ds = ds.repeat()
        ds = ds.shuffle(buffer_size=10 * total_batch_size, seed=0)
    else:
        if split.num_examples % total_batch_size != 0:
            raise ValueError(f'Test set size must be divisible by {total_batch_size}')

    def preprocess(example):
        image = _preprocess_image(example['image'], is_training)
        image = tf.transpose(image, (2, 0, 1))  # transpose HWC image to CHW format
        label = tf.cast(example['label'], tf.int32)
        return {'images': image, 'labels': label}

    ds = ds.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    for batch_size in reversed(batch_dims):
        ds = ds.batch(batch_size)

    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    yield from tfds.as_numpy(ds)


def normalize_image_for_view(image):
    """Normalizes dataset image into the format for viewing."""
    image *= np.reshape(STDDEV_RGB, (3, 1, 1))
    image += np.reshape(MEAN_RGB, (3, 1, 1))
    image = np.transpose(image, (1, 2, 0))
    return image.clip(0, 255).round().astype('uint8')


def _preprocess_image(
        image_bytes: tf.Tensor,
        is_training: bool,
) -> tf.Tensor:
    """Returns processed and resized images."""
    if is_training:
        image = _decode_and_random_crop(image_bytes)
        image = tf.image.random_flip_left_right(image)
    else:
        image = _decode_and_center_crop(image_bytes)
    assert image.dtype == tf.uint8
    # NOTE: Bicubic resize (1) casts uint8 to float32 and (2) resizes without
    # clamping overshoots. This means values returned will be outside the range
    # [0.0, 255.0].
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE],
                            tf.image.ResizeMethod.BICUBIC)
    image = _normalize_image(image)
    return image


def _normalize_image(image: tf.Tensor) -> tf.Tensor:
    """Normalize the image to zero mean and unit variance."""
    image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
    image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
    return image


def _distorted_bounding_box_crop(
        image_bytes: tf.Tensor,
        jpeg_shape: tf.Tensor,
        bbox: tf.Tensor,
        min_object_covered: float,
        aspect_ratio_range: Tuple[float, float],
        area_range: Tuple[float, float],
        max_attempts: int,
) -> tf.Tensor:
    """Generates cropped_image using one of the bboxes randomly distorted."""
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        jpeg_shape,
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)

    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    return image


def _decode_and_random_crop(image_bytes: tf.Tensor) -> tf.Tensor:
    """Make a random crop of image."""
    jpeg_shape = tf.image.extract_jpeg_shape(image_bytes)
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    image = _distorted_bounding_box_crop(
        image_bytes,
        jpeg_shape=jpeg_shape,
        bbox=bbox,
        min_object_covered=0.1,
        aspect_ratio_range=(3 / 4, 4 / 3),
        area_range=(0.08, 1.0),
        max_attempts=10)
    if tf.reduce_all(tf.equal(jpeg_shape, tf.shape(image))):
        # If the random crop failed fall back to center crop.
        image = _decode_and_center_crop(image_bytes, jpeg_shape)
    return image


def _decode_and_center_crop(
        image_bytes: tf.Tensor,
        jpeg_shape: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """Crops to center of image with padding then scales."""
    if jpeg_shape is None:
        jpeg_shape = tf.image.extract_jpeg_shape(image_bytes)
    image_height = jpeg_shape[0]
    image_width = jpeg_shape[1]

    padded_center_crop_size = tf.cast(
        ((IMAGE_SIZE / (IMAGE_SIZE + IMAGE_PADDING_FOR_CROP)) *
         tf.cast(tf.minimum(image_height, image_width), tf.float32)), tf.int32)

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    crop_window = tf.stack([offset_height, offset_width,
                            padded_center_crop_size, padded_center_crop_size])
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    return image
