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

__all__ = ['celu', 'elu', 'leaky_relu', 'log_sigmoid', 'log_softmax', 'logsumexp', 'relu',
           'selu', 'sigmoid', 'softmax', 'softplus', 'tanh']

import jax.nn
import jax.scipy.special
from jax import lax

from objax.typing import JaxArray

celu = jax.nn.celu
elu = jax.nn.elu
leaky_relu = jax.nn.leaky_relu
log_sigmoid = jax.nn.log_sigmoid
log_softmax = jax.nn.log_softmax
logsumexp = jax.scipy.special.logsumexp
selu = jax.nn.selu
sigmoid = jax.nn.sigmoid
softmax = jax.nn.softmax
softplus = jax.nn.softplus
tanh = lax.tanh


# Have to redefine relu since jax.nn.relu isn't pickable.
def relu(x: JaxArray) -> JaxArray:
    """Rectified linear unit activation function.

    Args:
        x: input tensor.

    Returns:
        tensor with the element-wise output relu(x) = max(x, 0).
    """
    return jax.nn.relu(x)
