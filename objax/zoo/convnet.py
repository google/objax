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

import jax

import objax
from objax.typing import JaxArray


class ConvNet(objax.nn.Sequential):
    """ConvNet implementation."""

    @staticmethod
    def _mean_reduce(x: JaxArray) -> JaxArray:
        return x.mean((2, 3))

    def __init__(self, nin, nclass, scales, filters, filters_max,
                 pooling=objax.functional.max_pool_2d, **kwargs):
        """Creates ConvNet instance.

        Args:
            nin: number of channels in the input image.
            nclass: number of output classes.
            scales: number of pooling layers, each of which reduces spatial dimension by 2.
            filters: base number of convolution filters.
                     Number of convolution filters is increased by 2 every scale until it reaches filters_max.
            filters_max: maximum number of filters.
            pooling: type of pooling layer.
        """
        del kwargs

        def nf(scale):
            return min(filters_max, filters << scale)

        ops = [objax.nn.Conv2D(nin, nf(0), 3), objax.functional.leaky_relu]
        for i in range(scales):
            ops.extend([objax.nn.Conv2D(nf(i), nf(i), 3), objax.functional.leaky_relu,
                        objax.nn.Conv2D(nf(i), nf(i + 1), 3), objax.functional.leaky_relu,
                        jax.partial(pooling, size=2, strides=2)])
        ops.extend([objax.nn.Conv2D(nf(scales), nclass, 3), self._mean_reduce])
        super().__init__(ops)
