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

__all__ = ['Adam']

from typing import List

from jax import numpy as jn

from objax import functional
from objax.module import Module, ModuleList
from objax.typing import JaxArray
from objax.variable import TrainRef, StateVar, TrainVar, VarCollection


class Adam(Module):
    """Adam optimizer."""

    def __init__(self, vc: VarCollection, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        """Constructor for Adam optimizer class.
        
        Args:
            vc: collection of variables to optimize.
            beta1: value of Adam's beta1 hyperparameter. Defaults to 0.9.
            beta2: value of Adam's beta2 hyperparameter. Defaults to 0.999.
            eps: value of Adam's epsilon hyperparameter. Defaults to 1e-8.
        """
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step = StateVar(jn.array(0, jn.uint32), reduce=lambda x: x[0])
        self.train_vars = ModuleList(TrainRef(x) for x in vc.subset(TrainVar))
        self.m = ModuleList(StateVar(jn.zeros_like(x.value)) for x in self.train_vars)
        self.v = ModuleList(StateVar(jn.zeros_like(x.value)) for x in self.train_vars)

    def __call__(self, lr: float, grads: List[JaxArray]):
        """Updates variables and other state based on Adam algorithm.
        
        Args:
            lr: the learning rate.
            grads: the gradients to apply.
        """
        assert len(grads) == len(self.train_vars), 'Expecting as many gradients as trainable variables'
        self.step.value += 1
        lr *= jn.sqrt(1 - self.beta2 ** self.step.value) / (1 - self.beta1 ** self.step.value)
        for g, p, m, v in zip(grads, self.train_vars, self.m, self.v):
            m.value += (1 - self.beta1) * (g - m.value)
            v.value += (1 - self.beta2) * (g ** 2 - v.value)
            p.value -= lr * m.value * functional.rsqrt(v.value + self.eps)
