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

import unittest

from jax import numpy as jn

import objax


class SampleModule(objax.Module):

    def __init__(self, nin, nout):
        self.v1 = objax.TrainVar(jn.ones((nin, nout)))
        self.var_list = [
            objax.TrainVar(jn.zeros(nin)),
            objax.TrainVar(jn.zeros(nout)),
        ]

    def __call__(self, x):
        return jn.dot(x, self.v1.value)


class TestModule(unittest.TestCase):
    """Unit tests for classes from module.py."""

    def test_module(self):
        """Unit test for objax.Module."""
        m = SampleModule(3, 5)
        self.assertIsInstance(m.vars(), objax.VarCollection)
        vars_list = list(m.vars())
        # Note that m.var_list is not in m.vars()
        self.assertEqual(len(vars_list), 1)
        self.assertIs(vars_list[0], m.v1)

    def test_module_list(self):
        """Unit test for objax.ModuleList."""
        m1 = SampleModule(3, 5)
        m2 = SampleModule(5, 7)
        module_list = objax.ModuleList([m1, m2])
        vars_list = list(module_list.vars())
        self.assertEqual(len(vars_list), 2)
        self.assertIs(vars_list[0], m1.v1)
        self.assertIs(vars_list[1], m2.v1)

    def test_module_wrapper(self):
        """Unit test for objax.ModuleWrapper."""
        m1 = SampleModule(3, 5)
        m2 = SampleModule(5, 7)
        var_collection = objax.ModuleList([m1, m2]).vars()
        module_wrapper = objax.ModuleWrapper(var_collection)
        module_wrapper_vars = module_wrapper.vars()
        self.assertEqual(len(var_collection), len(module_wrapper_vars))
        for k, v in var_collection.items():
            new_key = f'({module_wrapper.__class__.__name__}){k}'
            self.assertIn(new_key, module_wrapper_vars)
            self.assertIs(module_wrapper_vars[new_key], v)


if __name__ == '__main__':
    unittest.main()
