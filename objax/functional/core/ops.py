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


__all__ = ['dynamic_slice', 'flatten', 'interpolate', 'one_hot', 'pad', 'rsqrt', 'scan', 'stop_gradient',
           'top_k', 'upsample_2d', 'upscale_nn']

from typing import Union, Tuple

import jax.nn
from jax import numpy as jn, lax

from objax import util
from objax.constants import Interpolate
from objax.typing import JaxArray

dynamic_slice = lax.dynamic_slice
one_hot = jax.nn.one_hot
pad = jn.pad
scan = lax.scan
stop_gradient = lax.stop_gradient
top_k = lax.top_k  # Current code doesn't work with gradient.
rsqrt = lax.rsqrt


def flatten(x: JaxArray) -> JaxArray:
    """Flattens input tensor to a 2D tensor.

    Args:
        x: input tensor with dimensions (n_1, n_2, ..., n_k)

    Returns:
        The input tensor reshaped to two dimensions (n_1, n_prod),
        where n_prod is equal to the product of n_2 to n_k.
    """
    return x.reshape([x.shape[0], -1])


def interpolate(input: JaxArray,
                size: Union[int, Tuple[int, ...]] = None,
                scale_factor: Union[int, Tuple[int, ...]] = None,
                mode: Union[Interpolate, str] = Interpolate.BILINEAR) -> JaxArray:
    """
    Function to interpolate JaxArrays by size or scaling factor
    Args:
        input: input tensor
        size: int or tuple for output size
        scale_factor: int or tuple scaling factor for each dimention
        mode:str or Interpolate interpolation method e.g. ['bilinear', 'nearest']

    Returns:
        output : output JaxArray after interpolation
    """
    assert size or scale_factor, f'both size: {size} and scale_factor: {scale_factor} can not be None .'
    assert bool(size) ^ bool(scale_factor), f'either size or scale_factor must be none ' \
                                            f'scale: {size}, scale_factor: {scale_factor} .'
    input_shape = input.shape
    input_dim = len(input_shape)
    if scale_factor:
        if isinstance(scale_factor, int):
            size = (input_shape[0], *(jn.array(input_shape[1:]) * scale_factor))
        if isinstance(scale_factor, Tuple):
            output_dim = len(scale_factor)
            size = (*input_shape[:input_dim - output_dim],
                    *(jn.array(input_shape[input_dim - output_dim:]) * jn.array(scale_factor)))
    else:
        if isinstance(size, int):
            size = (*input_shape[:-1], size)
        if isinstance(size, Tuple):
            output_dim = len(size)
            assert input_dim >= output_dim, f'Number of dimensions of "{size}"' \
                                            f' must be < = to input shape"{input_shape}" '
            size = (*input_shape[:input_dim - output_dim], *size)
    output = jax.image.resize(input,
                              shape=size,
                              method=util.to_interpolate(mode))
    return output


def upsample_2d(x: JaxArray,
                scale: Union[Tuple[int, int], int],
                method: Union[Interpolate, str] = Interpolate.BILINEAR) -> JaxArray:
    """Function to upscale 2D images.

    Args:
        x: input tensor.
        scale: int or tuple scaling factor
        method: str or UpSample interpolation methods e.g. ['bilinear', 'nearest'].

    Returns:
        upscaled 2d image tensor
    """
    s = x.shape
    assert len(s) == 4, f'{s} must have 4 dimensions to be upsampled, or you can try interpolate function.'
    scale = util.to_tuple(scale, 2)
    y = jax.image.resize(x.transpose([0, 2, 3, 1]),
                         shape=(s[0], s[2] * scale[0], s[3] * scale[1], s[1]),
                         method=util.to_interpolate(method))
    return y.transpose([0, 3, 1, 2])


def upscale_nn(x: JaxArray, scale: int = 2) -> JaxArray:
    """Nearest neighbor upscale for image batches of shape (N, C, H, W).

    Args:
        x: input tensor of shape (N, C, H, W).
        scale: integer scaling factor.

    Returns:
        Output tensor of shape (N, C, H * scale, W * scale).
    """
    s = x.shape
    x = x.reshape(s[:2] + (s[2], 1, s[3], 1))
    x = jn.tile(x, (1, 1, 1, scale, 1, scale))
    return x.reshape(s[:2] + (scale * s[2], scale * s[3]))
