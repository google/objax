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

__all__ = ['dynamic_slice', 'pad', 'rsqrt', 'stop_gradient', 'top_k',
           'flatten', 'one_hot', 'upscale']

import jax.nn.functions as jnnf
from jax import numpy as jn, lax

from objax.typing import JaxArray

dynamic_slice = lax.dynamic_slice
one_hot = jnnf.one_hot
pad = jn.pad
stop_gradient = lax.stop_gradient
top_k = lax.top_k  # Current code doesn't work with gradient.
rsqrt = lax.rsqrt


def flatten(x: JaxArray) -> JaxArray:
    """Flattens input tensor to a 2D tensor.

    Args:
        x: input tensor with dimensions (n_1, n_2, ..., n_k)

    Returns:
        The input tensor reshaped to two dimensions (n_1, n_sum),
        where n_sum is equal to the sum of n_2 to n_k.
    """
    return x.reshape([x.shape[0], -1])



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
        x: input tensor of shape (N, C, H, W).

    Returns:
        Output tensor of shape (N, C, 2*H, 2*W).
    """
    s = x.shape
    x = x.reshape(s[:2] + (s[2], 1, s[3], 1))
    x = jn.tile(x, (1, 1, 1, 2, 1, 2))
    return x.reshape(s[:2] + (2 * s[2], 2 * s[3]))
