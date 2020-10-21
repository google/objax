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

"""Unittests for ObJAX JIT."""

import unittest

import jax.numpy as jn
from jax.core import ConcretizationTypeError

import objax
from objax.typing import JaxArray


class LinearArgs(objax.nn.Linear):
    def __call__(self, x: JaxArray, some_args: float) -> JaxArray:
        """Returns the results of applying the linear transformation to input x."""
        y = jn.dot(x, self.w.value) * some_args
        if self.b:
            y += self.b.value
        return y


class LinearTrain(objax.nn.Linear):
    def __call__(self, x: JaxArray, training: bool) -> JaxArray:
        """Returns the results of applying the linear transformation to input x."""
        y = jn.dot(x, self.w.value)
        if training:
            y = -y
        if self.b:
            y += self.b.value
        return y


class TestJit(unittest.TestCase):
    def test_on_linear(self):
        k = objax.nn.Linear(3, 3)
        kj = objax.Jit(k)
        x = objax.random.normal((64, 3))
        y1 = kj(x)
        k.w.assign(k.w.value + 1)
        y2 = kj(x)
        k.w.assign(k.w.value - 1)
        y3 = kj(x)
        self.assertAlmostEqual(((y1 - y3) ** 2).sum(), 0)
        self.assertNotEqual(((y1 - y2) ** 2).sum(), 0)

    def test_double_jit(self):
        k = objax.nn.Linear(3, 3)
        kj = objax.Jit(objax.Jit(k))
        x = objax.random.normal((64, 3))
        y1 = kj(x)
        k.w.assign(k.w.value + 1)
        y2 = kj(x)
        k.w.assign(k.w.value - 1)
        y3 = kj(x)
        self.assertAlmostEqual(((y1 - y3) ** 2).sum(), 0)
        self.assertNotEqual(((y1 - y2) ** 2).sum(), 0)

    def test_jit_kwargs(self):
        x = objax.random.normal((64, 3))
        kj = objax.Jit(LinearArgs(3, 3))
        y1 = kj(x, 1)
        y2 = kj(x, some_args=1)
        y3 = kj(x, some_args=2)
        self.assertEqual(y1.tolist(), y2.tolist())
        self.assertNotEqual(y1.tolist(), y3.tolist())
        kj = objax.Jit(LinearTrain(3, 3))
        with self.assertRaises(ConcretizationTypeError):
            kj(x, training=True)

    def test_trainvar_assign(self):
        m = objax.ModuleList([objax.TrainVar(jn.zeros(2))])

        def increase():
            m[0].assign(m[0].value + 1)
            return m[0].value

        jit_increase = objax.Jit(increase, m.vars())
        jit_increase()
        self.assertEqual(m[0].value.tolist(), [1., 1.])


if __name__ == '__main__':
    unittest.main()
