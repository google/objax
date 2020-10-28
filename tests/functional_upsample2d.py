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

"""Unittests for functional upsample operations."""

import unittest
import jax
import jax.numpy as jn
import numpy as np

import objax


def shaparange(s):
    return jn.arange(np.prod(s), dtype=np.float).reshape(s)


class TestUpsample(unittest.TestCase):
    def test_upsample2d(self):
        x = shaparange((2, 3, 10, 30))
        shape = x.shape
        for method in ['nearest', 'linear', 'bilinear', 'trilinear', 'triangle', 'cubic',
                       'bicubic', 'tricubic', 'lanczos3', 'lanczos5']:
            y = objax.functional.core.ops.upsample_2d(x, (2, 3), method)
            z = jax.image.resize(x.transpose([0, 2, 3, 1]),
                                 shape=(shape[0], shape[2] * 2, shape[3] * 3, shape[1]),
                                 method=method).transpose([0, 3, 1, 2])
            self.assertEqual(y.tolist(), z.tolist())


if __name__ == '__main__':
    unittest.main()
