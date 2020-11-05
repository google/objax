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

"""Random augment."""
from typing import Optional

import tensorflow as tf

from examples.fixmatch.libml.augment.randaugment import augment_ops

IMAGENET_AUG_OPS = [
    'AutoContrast', 'Equalize', 'Invert', 'Rotate', 'Posterize', 'Solarize',
    'Color', 'Contrast', 'Brightness', 'Sharpness', 'ShearX', 'ShearY',
    'TranslateX', 'TranslateY', 'SolarizeAdd', 'Identity',
]

# Levels in this file are assumed to be floats in [0, 1] range
# If you need quantization or integer levels, this should be controlled
# in client code.
MAX_LEVEL = 1.

# Constant which is used when computing translation argument from level
TRANSLATE_CONST = 100.


def _randomly_negate_tensor(tensor):
    """With 50% prob turn the tensor negative."""
    should_flip = tf.cast(tf.floor(tf.random.uniform([]) + 0.5), tf.bool)
    final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)
    return final_tensor


def _rotate_level_to_arg(level):
    level = (level / MAX_LEVEL) * 30.
    level = _randomly_negate_tensor(level)
    return level,


def _enhance_level_to_arg(level):
    return (level / MAX_LEVEL) * 1.8 + 0.1,


def _shear_level_to_arg(level):
    level = (level / MAX_LEVEL) * 0.3
    # Flip level to negative with 50% chance
    level = _randomly_negate_tensor(level)
    return level,


def _translate_level_to_arg(level):
    level = (level / MAX_LEVEL) * TRANSLATE_CONST
    # Flip level to negative with 50% chance
    level = _randomly_negate_tensor(level)
    return level,


def _posterize_level_to_arg(level):
    return int((level / MAX_LEVEL) * 4),


def _solarize_level_to_arg(level):
    return int((level / MAX_LEVEL) * 256),


def _solarize_add_level_to_arg(level):
    return int((level / MAX_LEVEL) * 110),


def _ignore_level_to_arg(level):
    del level
    return ()


def _divide_level_by_max_level_arg(level):
    return level / MAX_LEVEL,


LEVEL_TO_ARG = {
    'AutoContrast': _ignore_level_to_arg,
    'Equalize': _ignore_level_to_arg,
    'Invert': _ignore_level_to_arg,
    'Rotate': _rotate_level_to_arg,
    'Posterize': _posterize_level_to_arg,
    'Solarize': _solarize_level_to_arg,
    'SolarizeAdd': _solarize_add_level_to_arg,
    'Color': _enhance_level_to_arg,
    'Contrast': _enhance_level_to_arg,
    'Brightness': _enhance_level_to_arg,
    'Sharpness': _enhance_level_to_arg,
    'ShearX': _shear_level_to_arg,
    'ShearY': _shear_level_to_arg,
    'TranslateX': _translate_level_to_arg,
    'TranslateY': _translate_level_to_arg,
    'Identity': _ignore_level_to_arg,
    'Blur': _divide_level_by_max_level_arg,
    'Smooth': _divide_level_by_max_level_arg,
    'Rescale': _divide_level_by_max_level_arg,
}


class RandAugment:
    """Random augment with fixed magnitude."""

    def __init__(self,
                 n: int = 2,
                 p: Optional[float] = None,
                 magnitude: Optional[float] = None,
                 quantum: int = 10):
        """Create a RandAugment instance.

        Args:
          n: number of augmentations to perform per image.
          p: probability to apply an augmentation. If None then always apply.
          magnitude: default magnitude in range [0, 1]; magnitude will be chosen randomly when None.
          quantum: number of levels of quantization for the magnitude.
        """
        self.n = n
        self.p = p
        self.magnitude = magnitude
        self.quantum = quantum

    def _quantize(self):
        if self.magnitude is not None:
            return tf.convert_to_tensor(self.magnitude)
        if self.quantum is None:
            return tf.random.uniform(shape=[], dtype=tf.float32)
        else:
            level = tf.random.uniform(shape=[], maxval=self.quantum + 1, dtype=tf.int32)
            return tf.cast(level, tf.float32) / self.quantum

    def _pick_and_apply_augmentation(self, image):
        """Applies one level of augmentation to the image."""
        level = self._quantize()
        branch_fns = []
        for augment_op_name in IMAGENET_AUG_OPS:
            augment_fn = augment_ops.NAME_TO_FUNC[augment_op_name]
            level_to_args_fn = LEVEL_TO_ARG[augment_op_name]

            def _branch_fn(image=image, augment_fn=augment_fn, level_to_args_fn=level_to_args_fn):
                args = [image] + list(level_to_args_fn(level))
                return augment_fn(*args)

            branch_fns.append(_branch_fn)

        branch_index = tf.random.uniform(shape=[], maxval=len(branch_fns), dtype=tf.int32)
        aug_image = tf.switch_case(branch_index, branch_fns, default=lambda: image)
        if self.p is not None:
            return tf.cond(tf.random.uniform(shape=[], dtype=tf.float32) < self.p,
                           lambda: aug_image,
                           lambda: image)
        else:
            return aug_image

    def __call__(self, image):
        aug_image = image
        for _ in range(self.n):
            aug_image = self._pick_and_apply_augmentation(aug_image)
        return aug_image
