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

"""Unittests for module.py.

Note that some of the more complicated classes from module.py are tested in their own tests:
- objax.Jit is tested in tests/jit.py
- objax.Parallel is tested in tests/parallel.py
- objax.Vectorize is tested in tests/vectorize.py
"""
import functools
import inspect
import unittest
from typing import Tuple

from jax import numpy as jn

import objax
from objax.typing import JaxArray


class SimpleModule(objax.Module):
    def __init__(self, nin, nout):
        self.v1 = objax.TrainVar(jn.ones((nin, nout)))
        self.var_list = [objax.TrainVar(jn.zeros(nin)), objax.TrainVar(jn.zeros(nout))]

    def __call__(self, x):
        return jn.dot(x, self.v1.value)


class ComplexModule(objax.Module):
    def __init__(self, n):
        self.v = objax.TrainVar(jn.zeros(n))

    def __call__(self, x, training):
        if training:
            return self.v.value + x
        return self.v.value - x


class TestModule(unittest.TestCase):
    """Unit tests for classes from module.py."""

    def test_module(self):
        """Unit test for objax.Module."""
        m = SimpleModule(3, 5)
        self.assertIsInstance(m.vars(), objax.VarCollection)
        vars_list = list(m.vars())
        # Note that m.var_list is not in m.vars()
        self.assertEqual(len(vars_list), 1)
        self.assertIs(vars_list[0], m.v1)

    def test_module_list(self):
        """Unit test for objax.ModuleList."""
        m1 = SimpleModule(3, 5)
        m2 = SimpleModule(5, 7)
        module_list = objax.ModuleList([m1, m2])
        vars_list = list(module_list.vars())
        self.assertEqual(len(vars_list), 2)
        self.assertIs(vars_list[0], m1.v1)
        self.assertIs(vars_list[1], m2.v1)

    def test_module_function(self):
        """Unit test for objax.Function."""
        m = ComplexModule(2)

        def my_func(x: JaxArray, y: JaxArray, *, named: int = 1) -> Tuple[JaxArray, JaxArray]:
            return m(x, training=True) - named, m(y, training=False) + named

        module_func = objax.Function(my_func, m.vars())
        self.assertEqual(module_func.vars(), m.vars('{my_func}.'))
        self.assertEqual(inspect.signature(module_func), inspect.signature(my_func))
        x = jn.arange(6).reshape((3, 2))
        self.assertEqual([v.tolist() for v in module_func(x, x + 1)],
                         [v.tolist() for v in my_func(x, x + 1)])
        self.assertEqual([v.tolist() for v in module_func(x, x + 1, named=2)],
                         [v.tolist() for v in my_func(x, x + 1, named=2)])
        # partial doesn't retain names.
        module_func = objax.Function(functools.partial(my_func, named=2), m.vars())
        self.assertEqual(module_func.vars(), m.vars())

    def test_module_function_decorator(self):
        """Unit test for objax.Function.with_args"""
        m = ComplexModule(2)

        @objax.Function.with_vars(m.vars())
        def my_func(x: JaxArray, y: JaxArray, *, named: int = 1) -> Tuple[JaxArray, JaxArray]:
            return m(x, training=True) - named, m(y, training=False) + named

        self.assertEqual(my_func.vars(), m.vars('{my_func}.'))
        self.assertEqual(inspect.signature(my_func),
                         inspect.Signature([inspect.Parameter('x', inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                                              annotation=JaxArray),
                                            inspect.Parameter('y', inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                                              annotation=JaxArray),
                                            inspect.Parameter('named', inspect.Parameter.KEYWORD_ONLY,
                                                              annotation=int, default=1)],
                                           return_annotation=Tuple[JaxArray, JaxArray]))

        x = jn.arange(6).reshape((3, 2))
        u, v = my_func(x, x + 1)
        self.assertEqual(u.tolist(), [[-1.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(v.tolist(), [[0.0, -1.0], [-2.0, -3.0], [-4.0, -5.0]])
        u, v = my_func(x, x + 1, named=2)
        self.assertEqual(u.tolist(), [[-2.0, -1.0], [0.0, 1.0], [2.0, 3.0]])
        self.assertEqual(v.tolist(), [[1.0, 0.0], [-1.0, -2.0], [-3.0, -4.0]])


if __name__ == '__main__':
    unittest.main()
