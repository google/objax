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

__all__ = ['nchw', 'nhwc', 'to_png']

import io
from typing import Union

import numpy as np
from PIL import Image

from objax.typing import JaxArray


def nchw(x: Union[np.ndarray, JaxArray]) -> Union[np.ndarray, JaxArray]:
    """Converts an array in (N,H,W,C) format to (N,C,H,W) format."""
    dims = list(range(x.ndim))
    dims.insert(-2, dims.pop())
    return x.transpose(dims)


def nhwc(x: Union[np.ndarray, JaxArray]) -> Union[np.ndarray, JaxArray]:
    """Converts an array in (N,C,H,W) format to (N,H,W,C) format."""
    dims = list(range(x.ndim))
    dims.append(dims.pop(-3))
    return x.transpose(dims)


def to_png(x: np.ndarray) -> bytes:
    """Converts numpy array in (C,H,W) format into PNG format."""
    if x.dtype in (np.float64, np.float32, np.float16):
        x = np.transpose((x + 1) * 127.5, [1, 2, 0]).clip(0, 255).round().astype('uint8')
    elif x.dtype != np.uint8:
        raise ValueError('Unsupported array type, expecting float or uint8', x.dtype)
    if x.shape[2] == 1:
        x = np.broadcast_to(x, x.shape[:2] + (3,))
    with io.BytesIO() as f:
        Image.fromarray(x).save(f, 'png')
        return f.getvalue()
