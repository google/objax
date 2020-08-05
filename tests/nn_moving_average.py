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

"""Unittests for MovingAverage and ExponentialMovingAverage Layer."""

import unittest

import jax.numpy as jn
import numpy as np

import objax


class TestMovingAverage(unittest.TestCase):

    def test_MovingAverage(self):
        """Test MovingAverage."""
        x1 = jn.array([[0, 1, 2]])
        x2 = jn.array([[0, 0, 0]])
        x3 = jn.array([[-3, -4, 5]])
        init_value = 100
        shape = x1.shape
        ma = objax.nn.MovingAverage(shape=shape, buffer_size=2, init_value=init_value)

        x_ma1 = ma(x1)
        x_ma2 = ma(x2)
        x_ma3 = ma(x3)

        np.testing.assert_allclose(x_ma1, np.array([[50, 50.5, 51]]))
        np.testing.assert_allclose(x_ma2, np.array([[0, 0.5, 1]]))
        np.testing.assert_allclose(x_ma3, np.array([[-1.5, -2, 2.5]]))

    def test_ExponentialMovingAverage(self):
        """Test ExponentialMovingAverage."""
        x1 = jn.array([[0, 1, 2]]) * 100
        x2 = jn.array([[-3, -4, 5]]) * 100
        init_value = 100
        shape = x1.shape
        ema = objax.nn.ExponentialMovingAverage(shape=shape, init_value=init_value, momentum=0.8)

        x_ema1 = ema(x1)
        x_ema2 = ema(x2)

        np.testing.assert_allclose(x_ema1, np.array([[80, 100, 120]]))
        np.testing.assert_allclose(x_ema2, np.array([[4, 0, 196]]))


if __name__ == '__main__':
    unittest.main()
