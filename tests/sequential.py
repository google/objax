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


class TestSequential(unittest.TestCase):
    def test_on_sequential_linear_relu(self):
        """
        Pass an input through a linear filter with 3 units followed by ReLU and
        test the shape and contents of the output.
        """

        # Define linear filter with 1 input channel and 3 output channels
        linear_filter = objax.nn.Linear(2, 3, use_bias=False)
        weights = objax.TrainVar(jn.array([[1., 2., 1.], [2., 1., 2.]]))
        linear_filter.w = weights
        sequential = objax.nn.Sequential([linear_filter,
                                          objax.functional.relu])

        # Define data and compute output response of linear filter
        data = jn.array([[1., -1.], [2., -2.]])
        features = sequential(data)
        expected_features = jn.array([[0., 1., 0.], [0., 2., 0.]])
        self.assertEqual(features.shape, (2, 3))
        self.assertTrue(jn.array_equal(features, expected_features))

    def test_on_sequential_relu_linear(self):
        """
        Pass an input through a linear filter with 3 units followed by ReLU and
        test the shape and contents of the output.
        """

        # Define linear filter with 1 input channel and 3 output channels
        linear_filter = objax.nn.Linear(2, 3, use_bias=False)
        weights = objax.TrainVar(jn.array([[1., 2., 1.], [2., 1., 2.]]))
        linear_filter.w = weights
        sequential = objax.nn.Sequential([objax.functional.relu,
                                          linear_filter])

        # Define data and compute output response of linear filter
        data = jn.array([[1., -1.], [2., -2.]])
        features = sequential(data)
        expected_features = jn.array([[1., 2., 1.], [2., 4., 2.]])
        self.assertEqual(features.shape, (2, 3))
        self.assertTrue(jn.array_equal(features, expected_features))


if __name__ == '__main__':
    unittest.main()
