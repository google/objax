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

__all__ = ['partial', 'pmax', 'pmean', 'pmin', 'psum']

from functools import partial

import jax
from jax import lax


def pmax(x: jax.interpreters.pxla.ShardedDeviceArray, axis_name: str = 'device'):
    """Compute a multi-device reduce max on x over the device axis axis_name."""
    return lax.pmax(x, axis_name)


def pmean(x: jax.interpreters.pxla.ShardedDeviceArray, axis_name: str = 'device'):
    """Compute a multi-device reduce mean on x over the device axis axis_name."""
    return lax.pmean(x, axis_name)


def pmin(x: jax.interpreters.pxla.ShardedDeviceArray, axis_name: str = 'device'):
    """Compute a multi-device reduce min on x over the device axis axis_name."""
    return lax.pmin(x, axis_name)


def psum(x: jax.interpreters.pxla.ShardedDeviceArray, axis_name: str = 'device'):
    """Compute a multi-device reduce sum on x over the device axis axis_name."""
    return lax.psum(x, axis_name)
