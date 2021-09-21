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

"""Unittests for recurrent layers."""

import unittest

import jax.numpy as jn

import objax


class TestLSTMCell(unittest.TestCase):
    def test_on_lstm_three_unit(self):
        """
        Pass an input through a lstm cell and test contents of the output.
        """
        lstm_cell = objax.nn.LSTMCell(1, 1, use_bias=False)
        weights_i2h = objax.TrainVar(jn.array([[1., 2., 1., -1]]))
        weights_h2h = objax.TrainVar(jn.array([[1., 3., 1., -2]]))
        lstm_cell.i2h.w = weights_i2h
        lstm_cell.h2h.w = weights_h2h

        input = jn.array([[1.], [2.]])
        h, c = lstm_cell(input)
        self.assertTrue(jn.array_equal(h, jn.array([[0.13597059], [0.08232221]])))
        self.assertTrue(jn.array_equal(c, jn.array([[0.55676997], [0.84911263]])))

        h, c = lstm_cell(input, (h, c))
        self.assertTrue(jn.array_equal(h, jn.array([[0.17726903], [0.09630815]])))
        self.assertTrue(jn.array_equal(c, jn.array([[1.1262282], [1.6991038]])))

    def test_on_lstm_invalid_input_shape(self):
        """
        Pass an input with a dimension different from the input dimension
        of the lstm cell and test if that raises an error.
        """
        lstm_cell = objax.nn.LSTMCell(2, 1)
        input = jn.array([[1.]])
        with self.assertRaises(AssertionError):
            lstm_cell(input)


if __name__ == '__main__':
    unittest.main()
