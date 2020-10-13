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

"""Unittests for functional pooling operations."""

import unittest

import jax.numpy as  jn
import numpy as np

import objax


def shaparange(s):
    return jn.arange(np.prod(s), dtype=np.float).reshape(s)


class TestPooling(unittest.TestCase):
    def test_average_pooling2d(self):
        x = shaparange((2, 3, 10, 30))
        y = objax.functional.average_pool_2d(x, size=5)
        z = x.reshape((2, 3, 2, 5, 6, 5)).mean((-3, -1))
        self.assertEqual(y.tolist(), z.tolist())
        y = objax.functional.average_pool_2d(x, size=5, strides=1)
        z = np.zeros((2, 3, 6, 26), dtype=np.float)
        for i in range(6):
            for j in range(26):
                z[:, :, i, j] = x[:, :, i:i + 5, j:j + 5].mean((-2, -1))
        self.assertEqual(y.tolist(), z.tolist())
        y = objax.functional.average_pool_2d(x, size=(2, 3))
        z = x.reshape((2, 3, 5, 2, 10, 3)).mean((-3, -1))
        self.assertEqual(y.tolist(), z.tolist())

    def test_max_pooling2d(self):
        x = shaparange((2, 3, 10, 30))
        y = objax.functional.max_pool_2d(x, size=5)
        z = x.reshape((2, 3, 2, 5, 6, 5)).max((-3, -1))
        self.assertEqual(y.tolist(), z.tolist())
        y = objax.functional.max_pool_2d(x, size=5, strides=1)
        z = np.zeros((2, 3, 6, 26), dtype=np.float)
        for i in range(6):
            for j in range(26):
                z[:, :, i, j] = x[:, :, i:i + 5, j:j + 5].max((-2, -1))
        self.assertEqual(y.tolist(), z.tolist())
        y = objax.functional.max_pool_2d(x, size=(2, 3))
        z = x.reshape((2, 3, 5, 2, 10, 3)).max((-3, -1))
        self.assertEqual(y.tolist(), z.tolist())

    def test_pooling2d_padding(self):
        x = shaparange((2, 3, 10, 30))
        y = objax.functional.average_pool_2d(x, size=5, padding=(2, 3))
        z = np.pad(x, ((0, 0), (0, 0), (2, 3), (2, 3))).reshape((2, 3, 3, 5, 7, 5)).mean((-3, -1))
        self.assertEqual(y.tolist(), z.tolist())
        y = objax.functional.max_pool_2d(x, size=5, padding=(2, 3))
        z = np.pad(x, ((0, 0), (0, 0), (2, 3), (2, 3))).reshape((2, 3, 3, 5, 7, 5)).max((-3, -1))
        self.assertEqual(y.tolist(), z.tolist())
        y = objax.functional.average_pool_2d(x, size=5, padding=((2, 3), (3, 2)))
        z = np.pad(x, ((0, 0), (0, 0), (2, 3), (3, 2))).reshape((2, 3, 3, 5, 7, 5)).mean((-3, -1))
        self.assertEqual(y.tolist(), z.tolist())
        y = objax.functional.max_pool_2d(x, size=5, padding=((2, 3), (3, 2)))
        z = np.pad(x, ((0, 0), (0, 0), (2, 3), (3, 2))).reshape((2, 3, 3, 5, 7, 5)).max((-3, -1))
        self.assertEqual(y.tolist(), z.tolist())
        y = objax.functional.average_pool_2d(x, size=2, padding=1)
        z = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1))).reshape((2, 3, 6, 2, 16, 2)).mean((-3, -1))
        self.assertEqual(y.tolist(), z.tolist())
        y = objax.functional.max_pool_2d(x, size=2, padding=1)
        z = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1))).reshape((2, 3, 6, 2, 16, 2)).max((-3, -1))
        self.assertEqual(y.tolist(), z.tolist())

    def test_space_batch(self):
        """Test batch_to_space2d and space_to_batch2d."""
        x = shaparange((2, 3, 10, 30))
        y = objax.functional.space_to_batch2d(x, size=5)
        z = objax.functional.batch_to_space2d(y, size=5)
        self.assertEqual(x.tolist(), z.tolist())
        self.assertEqual(y.shape, (50, 3, 2, 6))
        y = objax.functional.space_to_batch2d(x, size=(2, 3))
        z = objax.functional.batch_to_space2d(y, size=(2, 3))
        self.assertEqual(x.tolist(), z.tolist())
        self.assertEqual(y.shape, (12, 3, 5, 10))

    def test_space_channel(self):
        """Test channel_to_space2d and space_to_channel2d."""
        x = shaparange((2, 3, 10, 30))
        y = objax.functional.space_to_channel2d(x, size=5)
        z = objax.functional.channel_to_space2d(y, size=5)
        self.assertEqual(x.tolist(), z.tolist())
        self.assertEqual(y.shape, (2, 75, 2, 6))
        y = objax.functional.space_to_channel2d(x, size=(2, 3))
        z = objax.functional.channel_to_space2d(y, size=(2, 3))
        self.assertEqual(x.tolist(), z.tolist())
        self.assertEqual(y.shape, (2, 18, 5, 10))


if __name__ == '__main__':
    unittest.main()
