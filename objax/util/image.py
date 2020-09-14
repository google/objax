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

__all__ = ['from_file', 'image_grid', 'nchw', 'nhwc', 'normalize_to_uint8', 'normalize_to_unit_float', 'to_png']

import io
from typing import Union, BinaryIO, IO

import jax.numpy as jn
import numpy as np
from PIL import Image

from objax.typing import JaxArray


def from_file(file: Union[str, IO[BinaryIO]]) -> np.ndarray:
    """Read an image from a file, convert it RGB and return it as an array.

    Args:
        file: filename or python file handle of the input file.

    Return:
        3D numpy array (C, H, W) normalized with normalize_to_unit_float.
    """
    image = np.asarray(Image.open(file).convert('RGB'))
    return normalize_to_unit_float(image.transpose((2, 0, 1)))


def image_grid(image: np.ndarray) -> np.ndarray:
    """Rearrange array of images (nh, hw, c, h, w) into image grid in a single image (c, nh * h, nh * w)."""
    s = image.shape
    return image.transpose([2, 0, 3, 1, 4]).reshape([s[2], s[3] * s[0], s[4] * s[1]])


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


def normalize_to_uint8(x: Union[np.ndarray, JaxArray]) -> Union[np.ndarray, JaxArray]:
    """Map a float image in [1/256-1, 1-1/256] to uint8 {0, 1, ..., 255}."""
    return (128 * (x + (1 - 1 / 256))).clip(0, 255).round().astype('uint8')


def normalize_to_unit_float(x: Union[np.ndarray, JaxArray]) -> Union[np.ndarray, JaxArray]:
    """Map an uint8 image in {0, 1, ..., 255} to float interval [1/256-1, 1-1/256]."""
    return x * (1 / 128) + (1 / 256 - 1)


def to_png(x: Union[np.ndarray, JaxArray]) -> bytes:
    """Converts numpy array in (C,H,W) format into PNG format."""
    if isinstance(x, jn.ndarray):
        x = np.array(x)
    if x.dtype in (np.float64, np.float32, np.float16):
        x = np.transpose(normalize_to_uint8(x), (1, 2, 0))
    elif x.dtype != np.uint8:
        raise ValueError('Unsupported array type, expecting float or uint8', x.dtype)
    if x.shape[2] == 1:
        x = np.broadcast_to(x, x.shape[:2] + (3,))
    with io.BytesIO() as f:
        Image.fromarray(x).save(f, 'png')
        return f.getvalue()
