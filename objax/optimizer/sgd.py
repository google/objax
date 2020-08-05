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

__all__ = ['SGD']

from typing import List

from objax.module import Module, ModuleList
from objax.typing import JaxArray
from objax.variable import TrainRef, TrainVar, VarCollection


class SGD(Module):
    """Stochastic Gradient Descent (SGD) optimizer."""

    def __init__(self, vc: VarCollection):
        """Constructor for SGD optimizer.
        
        Args:
            vc: collection of variables to optimize.
        """
        self.train_vars = ModuleList(TrainRef(x) for x in vc.subset(TrainVar))

    def __call__(self, lr: float, grads: List[JaxArray]):
        """Updates variables based on SGD algorithm.
        
        Args:
            lr: the learning rate.
            grads: the gradients to apply.
        """
        assert len(grads) == len(self.train_vars), 'Expecting as many gradients as trainable variables'
        for g, p in zip(grads, self.train_vars):
            p.value -= lr * g
