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

__all__ = ['average_pool_2d', 'batch_to_space2d', 'channel_to_space2d', 'max_pool_2d', 'space_to_batch2d',
           'space_to_channel2d']

from typing import Union, Tuple, Optional

import numpy as np
from jax import numpy as jn, lax

from objax.constants import ConvPadding
from objax.typing import JaxArray, ConvPaddingInt
from objax.util import to_padding, to_tuple


def average_pool_2d(x: JaxArray,
                    size: Union[Tuple[int, int], int] = 2,
                    strides: Optional[Union[Tuple[int, int], int]] = None,
                    padding: Union[ConvPadding, str, ConvPaddingInt] = ConvPadding.VALID) -> JaxArray:
    """Applies average pooling using a square 2D filter.

    Args:
        x: input tensor of shape (N, C, H, W).
        size: size of pooling filter.
        strides: stride step, use size when stride is none (default).
        padding: padding of the input tensor, either Padding.SAME or Padding.VALID or numerical values.

    Returns:
        output tensor of shape (N, C, H, W).
    """
    size = to_tuple(size, 2)
    strides = to_tuple(strides, 2) if strides else size
    padding = to_padding(padding, 2)
    if isinstance(padding, tuple):
        padding = ((0, 0), (0, 0)) + padding
    return lax.reduce_window(x, 0, lax.add, (1, 1) + size, (1, 1) + strides, padding=padding) / np.prod(size)


def batch_to_space2d(x: JaxArray, size: Union[Tuple[int, int], int] = 2) -> JaxArray:
    """Transfer batch dimension N into spatial dimensions (H, W).

    Args:
        x: input tensor of shape (N, C, H, W).
        size: size of spatial area.

    Returns:
        output tensor of shape (N // (size[0] * size[1]), C, H * size[0], W * size[1]).
    """
    size = to_tuple(size, 2)
    s = x.shape
    y = x.reshape((-1, size[0], size[1], s[1], s[2], s[3]))
    y = y.transpose((0, 3, 4, 1, 5, 2))
    return y.reshape((s[0] // (size[0] * size[1]), s[1], s[2] * size[0], s[3] * size[1]))


def channel_to_space2d(x: JaxArray, size: Union[Tuple[int, int], int] = 2) -> JaxArray:
    """Transfer channel dimension C into spatial dimensions (H, W).

    Args:
        x: input tensor of shape (N, C, H, W).
        size: size of spatial area.

    Returns:
        output tensor of shape (N, C // (size[0] * size[1]), H * size[0], W * size[1]).
    """
    size = to_tuple(size, 2)
    s = x.shape
    y = x.reshape((s[0], -1, size[0], size[1], s[2], s[3]))
    y = y.transpose((0, 1, 4, 2, 5, 3))
    return y.reshape((s[0], s[1] // (size[0] * size[1]), s[2] * size[0], s[3] * size[1]))


def max_pool_2d(x: JaxArray,
                size: Union[Tuple[int, int], int] = 2,
                strides: Optional[Union[Tuple[int, int], int]] = None,
                padding: Union[ConvPadding, str, ConvPaddingInt] = ConvPadding.VALID) -> JaxArray:
    """Applies max pooling using a square 2D filter.

    Args:
        x: input tensor of shape (N, C, H, W).
        size: size of pooling filter.
        strides: stride step, use size when stride is none (default).
        padding: padding of the input tensor, either Padding.SAME or Padding.VALID or numerical values.

    Returns:
        output tensor of shape (N, C, H, W).
    """
    size = to_tuple(size, 2)
    strides = to_tuple(strides, 2) if strides else size
    padding = to_padding(padding, 2)
    if isinstance(padding, tuple):
        padding = ((0, 0), (0, 0)) + padding
    return lax.reduce_window(x, -jn.inf, lax.max, (1, 1) + size, (1, 1) + strides, padding=padding)


def space_to_batch2d(x: JaxArray, size: Union[Tuple[int, int], int] = 2) -> JaxArray:
    """Transfer spatial dimensions (H, W) into batch dimension N.

    Args:
        x: input tensor of shape (N, C, H, W).
        size: size of spatial area.

    Returns:
        output tensor of shape (N * size[0] * size[1]), C, H // size[0], W // size[1]).
    """
    size = to_tuple(size, 2)
    s = x.shape
    y = x.reshape((s[0], s[1], s[2] // size[0], size[0], s[3] // size[1], size[1]))
    y = y.transpose((0, 3, 5, 1, 2, 4))
    return y.reshape((s[0] * size[0] * size[1], s[1], s[2] // size[0], s[3] // size[1]))


def space_to_channel2d(x: JaxArray, size: Union[Tuple[int, int], int] = 2) -> JaxArray:
    """Transfer spatial dimensions (H, W) into channel dimension C.

    Args:
        x: input tensor of shape (N, C, H, W).
        size: size of spatial area.

    Returns:
        output tensor of shape (N, C * size[0] * size[1]), H // size[0], W // size[1]).
    """
    size = to_tuple(size, 2)
    s = x.shape
    y = x.reshape((s[0], s[1], s[2] // size[0], size[0], s[3] // size[1], size[1]))
    y = y.transpose((0, 1, 3, 5, 2, 4))
    return y.reshape((s[0], s[1] * size[0] * size[1], s[2] // size[0], s[3] // size[1]))
