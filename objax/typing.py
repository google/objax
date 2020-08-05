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

"""This module contains type declarations for Objax."""

__all__ = ['FileOrStr', 'JaxArray', 'JaxDType']

from typing import Union, IO, BinaryIO

import jax.numpy as jn
from jax.interpreters.pxla import ShardedDeviceArray

FileOrStr = Union[str, IO[BinaryIO]]
JaxArray = Union[jn.ndarray, jn.DeviceArray, ShardedDeviceArray]
JaxDType = Union[jn.complex64, jn.complex128, jn.bfloat16,
                 jn.float16, jn.float32, jn.float64,
                 jn.int8, jn.int16, jn.int32, jn.int64,
                 jn.uint8, jn.uint16, jn.uint32, jn.uint64]
