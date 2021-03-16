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

__all__ = ['LinearAnnealing', 'StepDecay']


import abc
from typing import List, Tuple, Union

import jax.numpy as jn


class Scheduler:
    def __init__(self,
                 base_lr: float = 1.0):
        """Constructs an instance for learning rate scheduler.

        Args:
            base_lr: base learning rate.
        """
        self.base_lr = base_lr

    @abc.abstractmethod
    def multiplier(self, step: float):
        """Returns learning rate multiplier w.r.t. certain schedule."""
        raise NotImplementedError

    def __call__(self, step: float):
        """Returns learning rate or multiplier at certain step.

        Args:
            step: number of training step. When 0, we use the step counter.

        Returns:
            learning rate when base_lr is provided; otherwise,
            only multiplier is returned.
        """
        return self.base_lr * self.multiplier(step)


class LinearAnnealing(Scheduler):
    def __init__(self,
                 max_step: float,
                 base_lr: float = 1.0,
                 is_cycle: bool = True,
                 min_lr: float = 0.0):
        """Constructs an instance for linear annealing learning rate scheduler.

        Args:
            max_step: maximum number of train step.
            base_lr: base learning rate.
            is_cycle: trigger cyclical learning rate multiplier when step
                exceeds max_step.
            min_lr: minimum learning rate at max_step.
        """
        super().__init__(base_lr=base_lr)
        assert base_lr >= min_lr, (
            'base_lr should be greater than or equal to min_lr.')
        self.max_step = max_step
        self.is_cycle = is_cycle
        self.min_lr_multiplier = min_lr / self.base_lr

    def multiplier(self, step: float):
        """Returns linear annealing learning rate multiplier."""

        # If is_cycle, we use the remainder of step; otherwise, we stop update.
        if self.is_cycle:
            step = jn.remainder(step, self.max_step)
        else:
            step = jn.minimum(step, self.max_step)

        return 1.0 - (step / self.max_step) * (
            1.0 - self.min_lr_multiplier)


class StepDecay(Scheduler):
    def __init__(self,
                 step_size: Union[float, List, Tuple],
                 base_lr: float = 1.0,
                 gamma: float = 0.1):
        """Constructs an instance for step decay learning rate scheduler.

        Args:
            step_size: number of train steps to reduce learning rate.
            base_lr: base learning rate.
            gamma: learning rate decay rate.
        """
        super().__init__(base_lr=base_lr)
        self.gamma = gamma
        self.step_size = step_size

    def multiplier(self, step: float):
        """Returns step decay learning rate multiplier."""
        if isinstance(self.step_size, (tuple, list)):
            exponent = jn.sum(jn.greater_equal(step, jn.array(self.step_size)))
        else:
            exponent = step // self.step_size
        return self.gamma ** exponent
