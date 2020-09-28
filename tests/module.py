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

import inspect
import unittest

from jax import numpy as jn
import objax
import numpy as np


class SampleModule(objax.Module):

    def __init__(self, nin, nout):
        self.v1 = objax.TrainVar(jn.ones((nin, nout)))
        self.var_list = [
            objax.TrainVar(jn.zeros(nin)),
            objax.TrainVar(jn.zeros(nout)),
        ]

    def __call__(self, x):
        return jn.dot(x, self.v1.value)


class OpWithArg(objax.Module):

    def __call__(self, x, some_arg):
        return x + some_arg


class ModelBlockWithArg(objax.Module):

    def __init__(self):
        self.seq = objax.nn.Sequential([OpWithArg(), OpWithArg()])
        self.op1 = OpWithArg()

    def __call__(self, x, some_arg):
        # without forced args equivalent to (x + some_arg*3)
        y = self.seq(x, some_arg=some_arg)
        return self.op1(y, some_arg)


class ModelWithArg(objax.Module):

    def __init__(self):
        self.block1 = ModelBlockWithArg()
        self.block2 = ModelBlockWithArg()
        self.op1 = OpWithArg()

    def __call__(self, x, some_arg):
        # without forces args equivalent to (x + some_arg*7)
        y1 = self.block1(x, some_arg=some_arg)
        y2 = self.block2(y1, some_arg)
        return self.op1(y2, some_arg=some_arg)


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

    def test_force_args(self):
        x = jn.array([1., 2.])
        model = ModelWithArg()
        np.testing.assert_almost_equal(model(x, some_arg=0.0), [1., 2.])
        np.testing.assert_almost_equal(model(x, some_arg=1.0), [8., 9.])
        # Ensure that can not use invalid argument in original module
        with self.assertRaises(TypeError):
            model(x, wrong_arg_name=0.0)
        with self.assertRaises(TypeError):
            model.block1.op1(x, wrong_arg_name=0.0)
        # Set forced args
        original_signature = inspect.signature(model.block1.op1)
        model.block1.op1 = objax.ForceArgs(model.block1.op1, some_arg=-1.0)
        self.assertEqual(original_signature, inspect.signature(model.block1.op1))
        np.testing.assert_almost_equal(model(x, some_arg=0.0), [0., 1.])
        np.testing.assert_almost_equal(model(x, some_arg=1.0), [6., 7.])
        # ForceArgs does not allow to pass invalid args
        with self.assertRaises(TypeError):
            model.block1.op1(x, wrong_arg_name=1.0)
        # Set forced args in a list
        model.block1.seq[0] = objax.ForceArgs(model.block1.seq[0], some_arg=-1.0)
        np.testing.assert_almost_equal(model(x, some_arg=0.0), [-1., 0.])
        np.testing.assert_almost_equal(model(x, some_arg=1.0), [4., 5.])
        # Set invalid arg in forced args
        model.block2.op1 = objax.ForceArgs(model.block2.op1, wrong_arg_name=1.0)
        with self.assertRaises(TypeError):
            model(x, some_arg=0.0)
        # Undo force args
        objax.ForceArgs.undo(model)
        np.testing.assert_almost_equal(model(x, some_arg=0.0), [1., 2.])
        np.testing.assert_almost_equal(model(x, some_arg=1.0), [8., 9.])


if __name__ == '__main__':
    unittest.main()
