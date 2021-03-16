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
    methods = ['nearest', 'linear', 'bilinear', 'trilinear', 'triangle', 'cubic',
               'bicubic', 'tricubic', 'lanczos3', 'lanczos5']

    def test_upsample2d(self):
        x = shaparange((2, 3, 10, 30))
        shape = x.shape
        for method in self.methods:
            y = objax.functional.core.ops.upsample_2d(x, (2, 3), method)
            output = jax.image.resize(x.transpose([0, 2, 3, 1]),
                                      shape=(shape[0], shape[2] * 2, shape[3] * 3, shape[1]),
                                      method=method).transpose([0, 3, 1, 2])
            self.assertEqual(y.tolist(), output.tolist())

    def test_interpolate(self):
        x = shaparange((1, 3, 2, 3, 10, 30))
        shape = x.shape
        for method in self.methods:
            output = 2
            y = objax.functional.core.ops.interpolate(x, scale_factor=output, mode=method)
            self.assertEqual(y.shape, (shape[0], *(jn.array(shape[1:])) * output))
            output = (2, 2, 2)
            y = objax.functional.core.ops.interpolate(x, scale_factor=output, mode=method)
            self.assertEqual(y.shape, (*shape[:len(shape) - len(output)],
                                       *(jn.array(shape[len(shape) - len(output):]) * jn.array(output))))
            output = (2, 2, 2, 2)
            y = objax.functional.core.ops.interpolate(x, scale_factor=output, mode=method)
            self.assertEqual(y.shape, (*shape[:len(shape) - len(output)],
                                       *(jn.array(shape[len(shape) - len(output):]) * jn.array(output))))
            output = 2
            y = objax.functional.core.ops.interpolate(x, size=output, mode=method)
            self.assertEqual(y.shape, (*shape[:-1], output))
            output = (2, 2, 2)
            y = objax.functional.core.ops.interpolate(x, size=output, mode=method)
            self.assertEqual(y.shape, (*shape[:len(shape) - len(output)], *output))
            output = (2, 2, 2, 2)
            y = objax.functional.core.ops.interpolate(x, size=output, mode=method)
            self.assertEqual(y.shape, (*shape[:len(shape) - len(output)], *output))


if __name__ == '__main__':
    unittest.main()
