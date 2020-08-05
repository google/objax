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

__all__ = ['celu', 'dynamic_slice', 'elu', 'leaky_relu', 'log_sigmoid', 'log_softmax', 'logsumexp',
           'pad', 'rsqrt', 'selu', 'sigmoid', 'softmax', 'softplus', 'stop_gradient', 'tanh', 'top_k', 'relu',
           'average_pool_2d', 'flatten', 'max_pool_2d', 'one_hot', 'upscale']

from typing import Union, Tuple

import jax.nn.functions as jnnf
import jax.scipy.special
from jax import numpy as jn, lax

from objax.constants import ConvPadding
from objax.typing import JaxArray
from objax.util import to_tuple

celu = jnnf.celu
dynamic_slice = lax.dynamic_slice
elu = jnnf.elu
leaky_relu = jnnf.leaky_relu
log_sigmoid = jnnf.log_sigmoid
log_softmax = jnnf.log_softmax
logsumexp = jax.scipy.special.logsumexp
one_hot = jnnf.one_hot
pad = jn.pad
selu = jnnf.selu
sigmoid = jnnf.sigmoid
softmax = jnnf.softmax
softplus = jnnf.softplus
stop_gradient = lax.stop_gradient
tanh = lax.tanh
top_k = lax.top_k  # Current code doesn't work with gradient.
rsqrt = lax.rsqrt


def average_pool_2d(x: JaxArray,
                    size: Union[Tuple[int, int], int] = 2,
                    strides: Union[Tuple[int, int], int] = 2,
                    padding: ConvPadding = ConvPadding.VALID) -> JaxArray:
    """Applies average pooling using a square 2D filter.

    Args:
        x: input tensor of shape NCHW.
        size: size of pooling filter.
        strides: stride step.
        padding: type of padding used in pooling operation.

    Returns:
        Output tensor.
    """
    size = to_tuple(size, 2)
    strides = to_tuple(strides, 2)
    return lax.reduce_window(x, 0, lax.add, (1, 1) + size, (1, 1) + strides, padding=padding.value)


def flatten(x: JaxArray) -> JaxArray:
    """Flattens input tensor to a 2D tensor.

    Args:
        x: input tensor with dimensions (n_1, n_2, ..., n_k)

    Returns:
        The input tensor reshaped to two dimensions (n_1, n_sum),
        where n_sum is equal to the sum of n_2 to n_k.
    """
    return x.reshape([x.shape[0], -1])


def max_pool_2d(x: JaxArray,
                size: Union[Tuple[int, int], int] = 2,
                strides: Union[Tuple[int, int], int] = 2,
                padding: ConvPadding = ConvPadding.VALID) -> JaxArray:
    """Applies max pooling using a square 2D filter.

    Args:
        x: input tensor of shape NCHW.
        size: size of pooling filter.
        strides: stride step.
        padding: type of padding used in pooling operation.

    Returns:
        output tensor.
    """
    size = to_tuple(size, 2)
    strides = to_tuple(strides, 2)
    return lax.reduce_window(x, -jn.inf, lax.max, (1, 1) + size, (1, 1) + strides, padding=padding.value)


# Have to redefine relu since jnnf.relu isn't pickable.
def relu(x: JaxArray) -> JaxArray:
    """Rectified linear unit activation function.

    Args:
        x: input tensor.

    Returns:
        tensor with the element-wise output relu(x) = max(x, 0).
    """
    return jnnf.relu(x)


# Sample code for top_k with gradient support
# def top_k(x: jn.ndarray, k: int):
#     """Select the top k slices from the last dimension."""
#     bcast_idxs = jn.broadcast_to(jn.arange(x.shape[-1]), x.shape)
#     sorted_vals, sorted_idxs = lax.sort_key_val(x, bcast_idxs)
#     topk_vals = lax.slice_in_dim(sorted_vals, -k, sorted_vals.shape[-1], axis=-1)
#     topk_idxs = lax.slice_in_dim(sorted_idxs, -k, sorted_idxs.shape[-1], axis=-1)
#     topk_vals = jn.flip(topk_vals, axis=-1)
#     topk_idxs = jn.flip(topk_idxs, axis=-1)
#     return topk_vals, topk_idxs


def upscale(x: JaxArray) -> JaxArray:
    """Applies a 2x upscale of image batches of shape (N, C, H, W).

    Args:
        x: input tensor of shape NCHW.

    Returns:
        Input tensor where each each is upscaled by 2x.
    """
    s = x.shape
    x = x.reshape(s[:2] + (s[2], 1, s[3], 1))
    x = jn.tile(x, (1, 1, 1, 2, 1, 2))
    return x.reshape(s[:2] + (2 * s[2], 2 * s[3]))
