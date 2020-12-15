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

__all__ = ['DEFAULT_GENERATOR', 'Generator', 'randint', 'normal', 'truncated_normal', 'uniform']

from typing import Optional, Tuple

import jax.random as jr

from objax.module import Module
from objax.util import class_name
from objax.variable import RandomState, VarCollection


class Generator(Module):
    """Random number generator module."""

    def __init__(self, seed: int = 0):
        """Create a random key generator, seed is the random generator initial seed."""
        super().__init__()
        self.initial_seed = seed
        self._key: Optional[RandomState] = None

    @property
    def key(self):
        """The random generator state (a tensor of 2 int32)."""
        if self._key is None:
            self._key = RandomState(self.initial_seed)
        return self._key

    def seed(self, seed: int = 0):
        """Sets a new random generator seed."""
        self.initial_seed = seed
        if self._key is not None:
            self._key.seed(seed)

    def __call__(self):
        """Generate a new generator state."""
        return self.key.split(1)[0]

    def vars(self, scope: str = '') -> VarCollection:
        self.key  # Make sure the key is created before collecting the vars.
        return super().vars(scope)

    def __repr__(self):
        return f'{class_name(self)}(seed={self.initial_seed})'


DEFAULT_GENERATOR = Generator(0)


def normal(shape: Tuple[int, ...], *, mean: float = 0, stddev: float = 1, generator: Generator = DEFAULT_GENERATOR):
    """Returns a ``JaxArray`` of shape ``shape`` with random numbers from a normal distribution
    with mean ``mean`` and standard deviation ``stddev``."""
    return jr.normal(generator(), shape=shape) * stddev + mean


def randint(shape: Tuple[int, ...], low: int, high: int, generator: Generator = DEFAULT_GENERATOR):
    """Returns a ``JaxAarray`` of shape ``shape`` with random integers in {low, ..., high-1}."""
    return jr.randint(generator(), shape=shape, minval=low, maxval=high)


def truncated_normal(shape: Tuple[int, ...], *,
                     stddev: float = 1,
                     lower: float = -2,
                     upper: float = 2,
                     generator: Generator = DEFAULT_GENERATOR):
    """Returns a ``JaxArray`` of shape ``shape`` with random numbers from a normal distribution
    with mean 0 and standard deviation ``stddev`` truncated by (``lower``, ``upper``)."""
    return jr.truncated_normal(generator(), shape=shape, lower=lower, upper=upper) * stddev


def uniform(shape: Tuple[int, ...], generator: Generator = DEFAULT_GENERATOR):
    """Returns a ``JaxArray`` of shape ``shape`` with random numbers from a uniform distribution [0, 1]."""
    return jr.uniform(generator(), shape=shape)
