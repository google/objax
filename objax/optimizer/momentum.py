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

__all__ = ['Momentum']

from typing import List

from jax import numpy as jn

from objax.module import Module, ModuleList
from objax.variable import TrainRef, StateVar, TrainVar, VarCollection


class Momentum(Module):
    """Momentum optimizer."""

    def __init__(self, vc: VarCollection, momentum: float = 0.9, nesterov: bool = False):
        """Constructor for momentum optimizer class.
        
        Args:
            vc: collection of variables to optimize.
            momentum: the momentum hyperparameter.
            nesterov: bool indicating whether to use the Nesterov method.
        """
        self.momentum = momentum
        self.nesterov = nesterov
        self.train_vars = ModuleList(TrainRef(x) for x in vc.subset(TrainVar))
        self.m = ModuleList(StateVar(jn.zeros_like(x.value)) for x in self.train_vars)

    def __call__(self, lr: float, grads: List[jn.ndarray]):
        """Updates variables and other state based on momentum (or Nesterov) SGD.

        Args:
           lr: the learning rate.
           grads: the gradients to apply.
        """
        assert len(grads) == len(self.train_vars), 'Expecting as many gradients as trainable variables'
        if self.nesterov:
            for g, p, m in zip(grads, self.train_vars, self.m):
                m.value = g + self.momentum * m.value
                p.value -= lr * (g + self.momentum * m.value)
        else:
            for g, p, m in zip(grads, self.train_vars, self.m):
                m.value = g + self.momentum * m.value
                p.value -= lr * m.value
