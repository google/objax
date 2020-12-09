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
"""Unittests for GradGrad."""

import inspect
import unittest
from typing import Tuple, Dict, List

import jax
import jax.numpy as jn
import numpy as np
from scipy.stats.distributions import chi2

import objax
from objax.typing import JaxArray
from objax.zoo.dnnet import DNNet


class TestGrad(unittest.TestCase):
    def test_grad_grad_bn(self):
        """Test if gradient has the correct value for linear regression."""
        model = objax.nn.Sequential([objax.nn.Conv2D(nin=2, nout=4, k=1),
                                     objax.nn.BatchNorm2D(4)])

        data = jn.ones((1, 2, 10, 10))

        def loss(x):
            return jn.sum(model(x, training=True))

        g1_fn = objax.Grad(loss, {}, (0,))
        out1 = g1_fn(data)[0]
        self.assertEqual(out1.shape, (1, 2, 10, 10))

        g2_fn = objax.Grad(loss, {}, (0,))
        out2 = g2_fn(data)[0]
        self.assertEqual(out2.shape, (1, 2, 10, 10))

        np.testing.assert_allclose(out1, out2)


if __name__ == '__main__':
    unittest.main()
