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

import functools

import jax

import objax
from objax.typing import JaxArray


class ConvNet(objax.nn.Sequential):
    @staticmethod
    def _mean_reduce(x: JaxArray) -> JaxArray:
        return x.mean((2, 3))

    def __init__(self, nin, nclass, scales, filters, filters_max, **kwargs):
        del kwargs

        def nf(scale):
            return min(filters_max, filters << scale)

        ops = [objax.nn.Conv2D(nin, nf(0), 3), objax.functional.leaky_relu]
        for i in range(scales):
            ops.extend([objax.nn.Conv2D(nf(i), nf(i), 3), objax.functional.leaky_relu,
                        objax.nn.Conv2D(nf(i), nf(i + 1), 3), objax.functional.leaky_relu,
                        functools.partial(objax.functional.average_pool_2d, size=2, strides=2)])
        ops.extend([objax.nn.Conv2D(nf(scales), nclass, 3), self._mean_reduce])
        super().__init__(ops)
