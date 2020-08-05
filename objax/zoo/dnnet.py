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

from typing import Callable, Iterable

from objax.nn import Linear, Sequential


class DNNet(Sequential):
    """Deep neural network (MLP) implementation."""

    def __init__(self, layer_sizes: Iterable[int], activation: Callable):
        """Creates DNNet instance.

        Args:
            layer_sizes: number of neurons for each layer.
            activation: layer activation.
        """
        layer_sizes = list(layer_sizes)
        assert len(layer_sizes) >= 2
        ops = []
        for i in range(1, len(layer_sizes)):
            ops.extend([Linear(layer_sizes[i - 1], layer_sizes[i]), activation])
        super().__init__(ops)
