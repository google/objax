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

__all__ = ['LARS']

from typing import List

import jax.numpy as jn

from objax.module import Module, ModuleList
from objax.typing import JaxArray
from objax.variable import TrainRef, StateVar, TrainVar, VarCollection


class LARS(Module):
    """Layerwise adaptive rate scaling (LARS) optimizer.

    See https://arxiv.org/abs/1708.03888
    """

    def __init__(self, vc: VarCollection, beta: float = 0.9, wd: float = 0.0001,
                 tc: float = 0.001, eps: float = 0.00001):
        """Constructor for LARS optimizer.

        Args:
            vc: collection of variables to optimize.
        """
        self.beta = beta
        self.wd = wd
        self.tc = tc
        self.eps = eps
        self.train_vars = ModuleList(TrainRef(x) for x in vc.subset(TrainVar))
        self.m = ModuleList(StateVar(jn.zeros_like(x.value)) for x in self.train_vars)

    def __call__(self, lr: float, grads: List[JaxArray]):
        """Updates variables based on LARS algorithm.

        Args:
            lr: the learning rate.
            grads: the gradients to apply.
        """
        assert len(grads) == len(self.train_vars), 'Expecting as many gradients as trainable variables'

        train_vars_norm = jn.linalg.norm([jn.linalg.norm(tv.value) for tv in self.train_vars])
        grad_norm = jn.linalg.norm([jn.linalg.norm(g) for g in grads])
        trust_ratio = self.tc * train_vars_norm / (grad_norm + self.wd * train_vars_norm + self.eps)
        clipped_trust_ratio = jn.where(jn.logical_or(grad_norm == 0., train_vars_norm == 0.), 1., trust_ratio)
        scaled_lr = lr * clipped_trust_ratio

        for g, p, m in zip(grads, self.train_vars, self.m):
            if self.wd != 0:
                g += self.wd * p.value
            scaled_grad = scaled_lr * g
            new_momentum = self.beta * m.value + scaled_grad
            p.value -= new_momentum
            m.value = new_momentum
