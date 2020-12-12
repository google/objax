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

"""Unittests for Simple RNN"""

import unittest

import jax.numpy as jn

from objax.nn.layers import SimpleRNN
from objax.functional import one_hot
from objax.functional.core.activation import relu
from objax.nn.init import identity


class TestSimpleRNN(unittest.TestCase):

    def test_simple_rnn(self):
        nin = nout = 3
        batch_size = 1
        num_hiddens = 1
        model = SimpleRNN(num_hiddens, nin, nout, activation=relu, w_init=identity)

        X = jn.arange(batch_size)
        X_one_hot = one_hot(X, nin)

        Z, _ = model(X_one_hot)

        self.assertEqual(Z.shape, (batch_size, nout))
        self.assertTrue(jn.array_equal(Z, X_one_hot))

        # Test passing in an explicit initial state
        state = jn.array([[2.]])
        Z, _ = model(X_one_hot, state)
        self.assertTrue(jn.array_equal(Z, jn.array([[3., 0., 0.]])))


if __name__ == '__main__':
    unittest.main()
