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

__all__ = ['kl']

import jax.numpy as jn

from objax.typing import JaxArray


def kl(p: JaxArray, q: JaxArray, eps: float = 2 ** -17) -> JaxArray:
    """Calculates the Kullback-Leibler divergence between arrays p and q."""
    return p.dot(jn.log(p + eps) - jn.log(q + eps))
