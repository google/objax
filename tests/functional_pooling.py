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

import jax.numpy as jn
import numpy as np

import objax


def shaparange(s):
    return jn.arange(np.prod(s)).reshape(s)


class TestPooling(unittest.TestCase):
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
