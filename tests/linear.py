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

"""Unittests for Convolution Layer."""

import unittest

import jax.numpy as jn

import objax


class TestLinear(unittest.TestCase):
    def test_on_linear_three_unit(self):
        """
        Pass an input through a linear filter with 3 units and
        test the shape and contents of the output.
        """

        # Define linear filter with 1 input channel and 3 output channels
        linear_filter = objax.nn.Linear(1, 3, use_bias=False)
        weights = objax.TrainVar(jn.array([[1., 2., 1.]]))
        linear_filter.w = weights

        # Define data and compute output response of linear filter
        data = jn.array([[1.], [2.]])
        features = linear_filter(data)
        expected_features = jn.array([[1., 2., 1.], [2., 4., 2.]])
        self.assertEqual(features.shape, (2, 3))
        self.assertTrue(jn.array_equal(features, expected_features))

    def test_on_linear_three_unit_with_bias(self):
        """
        Pass an input through a linear filter with 3 units and bias
        test the shape and contents of the output.
        """

        # Define linear filter with 1 input channel and 3 output channels
        linear_filter = objax.nn.Linear(1, 3, use_bias=True)
        weights = objax.TrainVar(jn.array([[1., 2., 1.]]))
        bias = objax.TrainVar(jn.array([2., 1., 2.]))
        linear_filter.w = weights
        linear_filter.b = bias

        # Define data and compute output response of linear filter
        data = jn.array([[1.], [2.]])
        features = linear_filter(data)
        expected_features = jn.array([[3., 3., 3.], [4., 5., 4.]])
        self.assertEqual(features.shape, (2, 3))
        self.assertTrue(jn.array_equal(features, expected_features))


if __name__ == '__main__':
    unittest.main()
