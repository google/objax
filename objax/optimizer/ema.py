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

__all__ = ['ExponentialMovingAverage', 'ExponentialMovingAverageModule']

from typing import Callable, Tuple, List

import jax.numpy as jn

from objax.module import Module, ModuleList
from objax.typing import JaxArray
from objax.util import class_name
from objax.variable import RandomState, TrainRef, StateVar, TrainVar, VarCollection


class ExponentialMovingAverage(Module):
    """Maintains exponential moving averages for each variable from provided VarCollection."""

    def __init__(self, vc: VarCollection, momentum: float = 0.999, debias: bool = False, eps: float = 1e-6):
        """Creates ExponentialMovingAverage instance with given hyperparameters.

        Args:
            momentum: the decay factor for the moving average.
            debias: bool indicating whether to use initialization bias correction.
            eps: small adjustment to prevent division by zero.
        """
        self.momentum = momentum
        self.debias = debias
        self.eps = eps
        self.step = StateVar(jn.array(0, jn.uint32), reduce=lambda x: x[0])
        # Deduplicate variables and skip RandomState vars since they cannot be averaged.
        trainable, non_trainable = {}, {}  # Use dicts since they are ordered since python >= 3.6
        for v in vc:
            if isinstance(v, RandomState):
                continue
            if isinstance(v, TrainRef):
                v = v.ref
            if isinstance(v, TrainVar):
                trainable[id(v)] = v
            else:
                non_trainable[id(v)] = v
        self.refs = ModuleList(list(non_trainable.values()) + [TrainRef(v) for v in trainable.values()])
        self.m = ModuleList(StateVar(jn.zeros_like(x.value)) for x in self.refs)

    def __call__(self):
        """Updates the moving average."""
        self.step.value += 1
        for ref, m in zip(self.refs, self.m):
            m.value += (1 - self.momentum) * (ref.value - m.value)

    def refs_and_values(self) -> Tuple[VarCollection, List[JaxArray]]:
        """Returns the VarCollection of variables affected by Exponential Moving Average (EMA) and
        their corresponding EMA values."""
        if self.debias:
            step = self.step.value
            debias = 1 / (1 - (1 - self.eps) * self.momentum ** step)
            tensors = [m.value * debias for ref, m in zip(self.refs, self.m)]
        else:
            tensors = self.m.vars().tensors()
        return self.refs.vars(), tensors

    def replace_vars(self, f: Callable):
        """Returns a function that acts as f called when variables are replaced by their averages.

        Args:
            f: function to be called on the stored averages.

        Returns:
            A function that returns the output of calling f with stored variables replaced by
            their moving averages.
        """

        def wrap(*args, **kwargs):
            refs, new_values = self.refs_and_values()
            original_values = refs.tensors()
            refs.assign(new_values)
            try:
                return f(*args, **kwargs)
            finally:
                refs.assign(original_values)

        return wrap

    def __repr__(self):
        return f'{class_name(self)}(momentum={self.momentum}, debias={self.debias}, eps={self.eps})'


class ExponentialMovingAverageModule(Module):
    """Creates a module that uses the moving average weights of another module."""

    def __init__(self, module: Module, momentum: float = 0.999, debias: bool = False, eps: float = 1e-6):
        """Creates ExponentialMovingAverageModule instance with given hyperparameters.

        Args:
            module: a module for which to compute the moving average.
            momentum: the decay factor for the moving average.
            debias: bool indicating whether to use initialization bias correction.
            eps: small adjustment to prevent division by zero.
        """
        self.__wrapped__ = module
        self.ema = ExponentialMovingAverage(module.vars(), momentum=momentum, debias=debias, eps=eps)

    def __call__(self, *args, **kwargs):
        """Calls the original module with moving average weights."""
        return self.ema.replace_vars(self.__wrapped__)(*args, **kwargs)

    def update_ema(self):
        """Updates the moving average."""
        self.ema()

    def __repr__(self):
        return f'{class_name(self)}(momentum={self.ema.momentum}, debias={self.ema.debias}, eps={self.ema.eps})'
