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
from objax.util import class_name
from objax.variable import TrainRef, StateVar, TrainVar, VarCollection


class LARS(Module):
    """Layerwise adaptive rate scaling (LARS) optimizer.

    See https://arxiv.org/abs/1708.03888
    """

    def __init__(self, vc: VarCollection,
                 momentum: float = 0.9,
                 weight_decay: float = 1e-4,
                 tc: float = 1e-3,
                 eps: float = 1e-5):
        """Constructor for LARS optimizer.

        Args:
            vc: collection of variables to optimize.
            momentum: coefficient used for the moving average of the gradient.
            weight_decay: weight decay coefficient.
            tc: trust coefficient eta ( < 1) for trust ratio computation.
            eps: epsilon used for trust ratio computation.
        """
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.tc = tc
        self.eps = eps
        self.train_vars = ModuleList(TrainRef(x) for x in vc.subset(TrainVar))
        self.m = ModuleList(StateVar(jn.zeros_like(x.value)) for x in self.train_vars)

    def __call__(self, lr: float, grads: List[JaxArray]):
        """Updates variables based on LARS algorithm.

        Args:
            lr: learning rate. The LARS paper suggests using lr = lr_0 * (1 -t/T)**2,
            where t is the current epoch number and T the maximum number of epochs.
            grads: the gradients to apply.
        """
        assert len(grads) == len(self.train_vars), 'Expecting as many gradients as trainable variables'

        for g, p, m in zip(grads, self.train_vars, self.m):
            p_norm = jn.linalg.norm(p.value)
            g_norm = jn.linalg.norm(g)
            trust_ratio = self.tc * p_norm / (g_norm + self.weight_decay * p_norm + self.eps)
            local_lr = lr * jn.maximum(jn.logical_or(p_norm == 0, g_norm == 0), trust_ratio)
            m.value = self.momentum * m.value + local_lr * (g + self.weight_decay * p.value)
            p.value -= m.value

    def __repr__(self):
        return f'{class_name(self)}(momentum={self.momentum}, weight_decay={self.weight_decay}, ' \
               f'tc={self.tc}, eps={self.eps})'
