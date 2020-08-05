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

"""Unittests for functional.loss."""

import unittest

import jax.numpy as jn

import objax


class TestLoss(unittest.TestCase):
    def test_on_xe_logits_2d(self):
        """Test cross-entropy on logits."""
        x = jn.array([[0.58, 0.4, 0.02], [0.36, 0.27, 0.37], [0.46, 0.24, 0.3], [0.35, 0.29, 0.36]])
        e = objax.functional.loss.cross_entropy_logits(10 * x, x)
        gold = jn.array([0.9881458, 1.126976, 1.2800856, 1.1140614])
        self.assertAlmostEqual(jn.abs(e - gold).sum(), 0, delta=1e-12)

    def test_on_xe_logits_2d_broadcast(self):
        """Test cross-entropy on logits, broadcasted labels."""
        x = jn.array([[0.58, 0.4, 0.02], [0.36, 0.27, 0.37], [0.46, 0.24, 0.3], [0.35, 0.29, 0.36]])
        e = objax.functional.loss.cross_entropy_logits(10 * x, x[:1])
        gold = jn.array([0.9881458, 1.278976, 1.1840856, 1.2140613])
        self.assertAlmostEqual(jn.abs(e - gold).sum(), 0, delta=1e-12)

    def test_on_xe_logits_5d(self):
        """Test cross-entropy on logits in 5d."""
        x = jn.array([[0.58, 0.4, 0.02], [0.36, 0.27, 0.37], [0.46, 0.24, 0.3], [0.35, 0.29, 0.36]])
        x = x.reshape((2, 1, 1, 2, 3))
        e = objax.functional.loss.cross_entropy_logits(10 * x, x)
        gold = jn.array([0.9881458, 1.126976, 1.2800856, 1.1140614]).reshape((2, 1, 1, 2))
        self.assertAlmostEqual(jn.abs(e - gold).sum(), 0, delta=1e-12)

    def test_on_xe_logits_5d_broadcast(self):
        """Test cross-entropy on logits in 5d, broadcasted labels."""
        x = jn.array([[0.58, 0.4, 0.02], [0.36, 0.27, 0.37], [0.46, 0.24, 0.3], [0.35, 0.29, 0.36]])
        x = x.reshape((2, 1, 1, 2, 3))
        e = objax.functional.loss.cross_entropy_logits(10 * x, x[:1])
        gold = jn.array([[[[0.9881458, 1.126976]]], [[[1.1840856, 1.1010613]]]])
        self.assertAlmostEqual(jn.abs(e - gold).sum(), 0, delta=1e-12)
        e = objax.functional.loss.cross_entropy_logits(10 * x, x[:, :, :, :1])
        gold = jn.array([[[[0.9881458, 1.278976]]], [[[1.2800856, 1.0900612]]]])
        self.assertAlmostEqual(jn.abs(e - gold).sum(), 0, delta=1e-12)

    def test_on_xe_logits_sparse(self):
        """Test cross-entropy on logits with sparse labels."""
        x = jn.array([[0.58, 0.4, 0.02], [0.36, 0.27, 0.37], [0.46, 0.24, 0.3], [0.35, 0.29, 0.36]])
        y = jn.array([0, 1, 2, 1], dtype=jn.uint32)
        e = objax.functional.loss.cross_entropy_logits_sparse(10 * x, y)
        gold = jn.array([0.15614605, 1.820976, 1.8720856, 1.5760615])
        self.assertAlmostEqual(jn.abs(e - gold).sum(), 0, delta=1e-12)


if __name__ == '__main__':
    unittest.main()
