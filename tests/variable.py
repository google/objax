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

"""Unittests for objax.variable."""

import unittest

import jax.numpy as jn

import objax


class TestVariable(unittest.TestCase):
    def test_train_var(self):
        """Test TrainVar behavior."""
        v = objax.TrainVar(jn.arange(6, dtype=jn.float32).reshape((2, 3)))
        self.assertEqual(v.value.shape, (2, 3))
        self.assertEqual(v.value.sum(), 15)
        with self.assertRaises(ValueError):
            v.value += 1
        v.assign(v.value + 1)
        self.assertEqual(v.value.sum(), 21)
        v.reduce(v.value)
        self.assertEqual(v.value.shape, (3,))
        self.assertEqual(v.value.tolist(), [2.5, 3.5, 4.5])

    def test_state_var(self):
        """Test StateVar behavior."""
        v = objax.StateVar(jn.arange(6, dtype=jn.float32).reshape((2, 3)))
        self.assertEqual(v.value.shape, (2, 3))
        self.assertEqual(v.value.sum(), 15)
        v.value += 2
        self.assertEqual(v.value.sum(), 27)
        v.assign(v.value - 1)
        self.assertEqual(v.value.sum(), 21)
        v.reduce(v.value)
        self.assertEqual(v.value.shape, (3,))
        self.assertEqual(v.value.tolist(), [2.5, 3.5, 4.5])

    def test_state_ref(self):
        """Test TrainRef behavior."""
        tv = objax.TrainVar(jn.arange(6, dtype=jn.float32).reshape((2, 3)))
        v = objax.TrainRef(tv)
        for vi in (v, tv):
            self.assertEqual(vi.value.shape, (2, 3))
            self.assertEqual(vi.value.sum(), 15)
        v.value += 2
        self.assertEqual(v.value.sum(), 27)
        self.assertEqual(tv.value.sum(), 27)
        v.assign(v.value - 1)
        self.assertEqual(v.value.sum(), 21)
        self.assertEqual(tv.value.sum(), 21)
        v.reduce(v.value)
        for vi in (v, tv):
            self.assertEqual(vi.value.shape, (2, 3))
            self.assertEqual(vi.value.tolist(), [[1, 2, 3], [4, 5, 6]])
        tv.reduce(tv.value)
        for vi in (v, tv):
            self.assertEqual(vi.value.shape, (3,))
            self.assertEqual(vi.value.tolist(), [2.5, 3.5, 4.5])

    def test_random_state(self):
        """Test RandomState behavior."""
        v1 = objax.RandomState(0)
        v2 = objax.RandomState(1)
        v3 = objax.RandomState(0)
        self.assertEqual(v1.value.tolist(), [0, 0])
        self.assertEqual(v2.value.tolist(), [0, 1])
        s1 = v1.split(1)[0]
        self.assertEqual(s1.tolist(), [2718843009, 1272950319])
        self.assertEqual(v1.value.tolist(), [4146024105, 967050713])
        s2 = v1.split(1)[0]
        self.assertEqual(s2.tolist(), [1278412471, 2182328957])
        self.assertEqual(v1.value.tolist(), [2384771982, 3928867769])
        s1, s2 = v3.split(2)
        self.assertEqual(s1.tolist(), [3186719485, 3840466878])
        self.assertEqual(s2.tolist(), [2562233961, 1946702221])
        self.assertEqual(v3.value.tolist(), [2467461003, 428148500])

    def test_state_var_custom_reduce(self):
        """Test Custom StateVar reducing."""
        array = jn.array([[0, 4, 2], [3, 1, 5]], dtype=jn.float32)
        v = objax.StateVar(array, reduce=lambda x: x[0])
        v.reduce(v.value)
        self.assertEqual(v.value.tolist(), [0, 4, 2])
        v = objax.StateVar(array, reduce=lambda x: x.min(0))
        v.reduce(v.value)
        self.assertEqual(v.value.tolist(), [0, 1, 2])
        v = objax.StateVar(array, reduce=lambda x: x.max(0))
        v.reduce(v.value)
        self.assertEqual(v.value.tolist(), [3, 4, 5])

    def test_var_hierarchy(self):
        """Test variable hierarchy."""
        t = objax.TrainVar(jn.zeros(2))
        s = objax.StateVar(jn.zeros(2))
        r = objax.TrainRef(t)
        x = objax.RandomState(0)
        self.assertIsInstance(t, objax.TrainVar)
        self.assertIsInstance(t, objax.BaseVar)
        self.assertNotIsInstance(t, objax.BaseState)
        self.assertIsInstance(s, objax.BaseVar)
        self.assertIsInstance(s, objax.BaseState)
        self.assertNotIsInstance(s, objax.TrainVar)
        self.assertIsInstance(r, objax.BaseVar)
        self.assertIsInstance(r, objax.BaseState)
        self.assertNotIsInstance(r, objax.TrainVar)
        self.assertIsInstance(x, objax.BaseVar)
        self.assertIsInstance(x, objax.BaseState)
        self.assertIsInstance(x, objax.StateVar)
        self.assertNotIsInstance(x, objax.TrainVar)

    def test_assign_invalid_shape_and_type(self):
        """Test assigning invalid shape and type"""
        t = objax.TrainVar(jn.zeros(2))
        t.assign(jn.ones(2))
        with self.assertRaises(AssertionError):
            t.assign(jn.zeros(3))
        with self.assertRaises(AssertionError):
            t.assign([0, 1])
        t.assign(jn.ones(3), check=False)

    def test_replicate_shape_assert(self):
        """Test assigning invalid shape and type"""

        vc = objax.VarCollection({'var': objax.TrainVar(jn.zeros(5))})
        
        with vc.replicate():
            self.assertEqual(len(vc['var'].value.shape), 2)
            self.assertEqual(vc['var'].value.shape[-1], 5)


if __name__ == '__main__':
    unittest.main()
