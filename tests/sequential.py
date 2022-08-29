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

    def test_kwargs(self):
        """Test sequential on modules that take named inputs in kwargs."""

        class MyModule:
            def __init__(self):
                pass

            def __call__(self, x, some_param):
                return x + some_param

        seq = objax.nn.Sequential([MyModule(), MyModule()])
        self.assertEqual(seq(1, some_param=2), 5)
        with self.assertRaises(TypeError):
            seq(1)

    def test_variadic(self):
        """Test sequential on modules that take multiple inputs and have multiple outputs."""

        class MyModule:
            def __init__(self):
                pass

            def __call__(self, x, y):
                return x + y, x - y

        seq = objax.nn.Sequential([MyModule(), MyModule()])
        self.assertEqual(seq(1, 2), (2, 4))

    def test_slice(self):
        """Test sequential slices with variadic module."""

        class MyModule:
            def __init__(self, m):
                self.m = m

            def __call__(self, x, y):
                return self.m * x + y, self.m * x - y

        seq = objax.nn.Sequential([MyModule(2), MyModule(3)])
        self.assertEqual(seq(5, 7), (54, 48))
        self.assertEqual(seq[:1](5, 7), (17, 3))
        self.assertEqual(seq[1:](5, 7), (22, 8))

    def test_on_sequential_missing_argument(self):
        m = objax.nn.Sequential([objax.nn.Linear(2, 3), objax.nn.BatchNorm0D(3), objax.nn.Linear(3, 2)])
        x = jn.array([[1., -1.], [2., -2.]])
        msg = "missing 1 required positional argument: 'training'"
        try:
            m(x)
            assert False
        except TypeError as e:
            self.assertIn(msg, str(e))
        m.pop()
        try:
            m(x)
            assert False
        except TypeError as e:
            self.assertIn(msg, str(e))


if __name__ == '__main__':
    unittest.main()
