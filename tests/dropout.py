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

"""Unit Tests for Dropout layer."""

import unittest

import jax.numpy as jn

import objax
from objax import random


class TestDropout(unittest.TestCase):
    def test_on_dropout_0_5(self):
        """
        Pass an input through a Dropout layer
        that keeps half the input and test
        that half of the output values are zero.
        """

        drop_input = jn.array([[1., 2., 3., 4., 5., 6.]])
        keep = 0.5
        test_generator = random.DEFAULT_GENERATOR
        test_generator.seed(3)
        dropout_layer = objax.nn.Dropout(keep, test_generator)
        training = True
        drop_output = dropout_layer(drop_input, training)
        self.assertEqual(jn.count_nonzero(drop_output), 3)
        for index in range(drop_output.shape[1]):
            if drop_output[0][index] != 0:
                self.assertEqual(drop_output[0][index], drop_input[0][index] / keep)

    def test_on_dropout_two_dimension(self):
        """
        Pass a two dimensional input through a Dropout layer
        that keeps half the input and test
        that half of the output values are zero.
        """

        drop_input = jn.array([[1., 2., 3.], [5., 6., 7.]])
        keep = 0.5
        test_generator = random.DEFAULT_GENERATOR
        test_generator.seed(3)
        dropout_layer = objax.nn.Dropout(keep, test_generator)
        training = True
        drop_output = dropout_layer(drop_input, training)
        self.assertEqual(jn.count_nonzero(drop_output), 3)

    def test_on_dropout_1_0(self):
        """
        Pass an input through a Dropout layer
        that keeps all of the input and test
        that all of the output values are non-zero.
        """

        drop_input = jn.array([[1., 2., 3., 4., 5., 6.]])
        keep = 1.0
        dropout_layer = objax.nn.Dropout(keep)
        training = True
        drop_output = dropout_layer(drop_input, training)
        self.assertEqual(jn.count_nonzero(drop_output), 6)
        self.assertTrue(jn.array_equal(drop_input, drop_output))

    def test_on_dropout_0_0(self):
        """
        Pass an input through a Dropout layer
        that keeps none of the input and test
        that all of the output values are zero.
        """

        drop_input = jn.array([[1., 2., 3., 4., 5., 6.]])
        keep = 0.0
        test_generator = random.DEFAULT_GENERATOR
        test_generator.seed(1)
        dropout_layer = objax.nn.Dropout(keep, test_generator)
        training = True
        drop_output = dropout_layer(drop_input, training)
        self.assertEqual(jn.count_nonzero(drop_output), 0)

    def test_on_dropout_inference(self):
        """
        Pass an input to the Dropout layer when
        training is false and test that the output
        is equal to the input.
        """

        drop_input = jn.array([[1., 2., 3., 4., 5.]])
        dropout_layer = objax.nn.Dropout(0.5)
        training = False
        drop_output = dropout_layer(drop_input, training)
        self.assertTrue(jn.array_equal(drop_input, drop_output))


if __name__ == '__main__':
    unittest.main()
