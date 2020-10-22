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
           'flatten', 'one_hot', 'upscale_nn']

import jax.nn
from jax import numpy as jn, lax

from objax.typing import JaxArray

dynamic_slice = lax.dynamic_slice
one_hot = jax.nn.one_hot
pad = jn.pad
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
