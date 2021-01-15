# Copyright 2021 Google LLC
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


__all__ = []

from typing import Union, Sequence, Tuple, Callable, Optional

import jax.numpy as jn

from .typing import JaxArray
from .util import re_sign


def _pad(array: JaxArray,
         pad_width: Union[Sequence[Tuple[int, int]], Tuple[int, int], int],
         mode: Optional[Union[str, Callable]] = 'constant',
         *,
         stat_length: Optional[Union[Sequence[Tuple[int, int]], int]] = None,
         constant_values: Optional[Union[Sequence[Tuple[int, int]], int]] = 0,
         end_values: Optional[Union[Sequence[Tuple[int, int]], int]] = None,
         reflect_type: Optional[str] = None):
    # This is just to have a proper signature for jax.numpy.pad since the API, like in numpy, makes use of kwargs
    # and doesn't expose its arguments properly.
    pass


jn.pad = re_sign(_pad)(jn.pad)
