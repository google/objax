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

import numpy as np
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
        """Test replicating variable shapes does not assert"""
        vc = objax.VarCollection({'var': objax.TrainVar(jn.zeros(5))})
        with vc.replicate():
            self.assertEqual(len(vc['var'].value.shape), 2)
            self.assertEqual(vc['var'].value.shape[-1], 5)

    def test_jax_duck_typing_jax_api_one_float_arg(self):
        API_LIST = [
            'abs', 'absolute', 'all', 'alltrue', 'amax', 'amin', 'any', 'arccos', 'arccosh', 'arcsin', 'arcsinh',
            'arctan', 'arctanh', 'argmax', 'argmin', 'argsort', 'argwhere', 'around', 'asarray', 'average',
            'cbrt', 'cdouble', 'ceil', 'complex128', 'complex64', 'conj', 'conjugate',
            'corrcoef', 'cos', 'cosh', 'count_nonzero', 'csingle', 'cumprod', 'cumproduct', 'cumsum', 'deg2rad',
            'degrees', 'diag', 'double', 'empty_like', 'exp', 'exp2', 'expm1', 'fabs', 'fix', 'flatnonzero', 'float16',
            'float32', 'float64', 'floor', 'frexp', 'i0', 'int16', 'int32', 'int64', 'int8',
            'iscomplex', 'iscomplexobj', 'isfinite', 'isinf', 'isnan', 'isneginf', 'isposinf', 'isreal', 'isrealobj',
            'log', 'log10', 'log1p', 'log2', 'max', 'mean', 'median', 'min', 'modf', 'msort', 'nan_to_num',
            'nanargmax', 'nanargmin', 'nancumprod', 'nancumsum', 'nanmedian', 'nanmax', 'nanmean', 'nanmin', 'nanprod',
            'nanstd', 'nansum', 'nanvar', 'ndim', 'negative', 'nonzero', 'ones_like', 'positive', 'prod', 'product',
            'ptp', 'rad2deg', 'radians', 'ravel', 'reciprocal', 'rint', 'round', 'shape', 'sign', 'signbit', 'sin',
            'sinc', 'single', 'sinh', 'size', 'sometrue', 'sort', 'sort_complex', 'sqrt', 'square', 'squeeze', 'std',
            'sum', 'tan', 'tanh', 'transpose', 'trunc', 'uint16', 'uint32', 'uint64', 'uint8', 'vander', 'var',
            'zeros_like'
        ]
        v = objax.TrainVar(jn.array([1., 2., 3.], dtype=jn.float32))
        for name in API_LIST:
            fn = jn.__dict__[name]
            expected = fn(v.value)
            actual = fn(v)
            np.testing.assert_allclose(expected, actual)

    def test_jax_duck_typing_jax_api_two_float_arg(self):
        API_LIST = [
            'add', 'allclose', 'arctan2', 'array_equal', 'array_equiv', 'convolve', 'copysign', 'correlate', 'cross',
            'divide', 'divmod', 'dot', 'equal', 'float_power', 'floor_divide', 'fmax', 'fmin', 'fmod', 'greater',
            'greater_equal', 'heaviside', 'hypot', 'inner', 'isclose', 'kron', 'less', 'less_equal', 'logaddexp',
            'logaddexp2', 'matmul', 'maximum', 'minimum', 'mod', 'multiply', 'nextafter', 'not_equal', 'outer',
            'polyadd', 'polysub', 'power', 'remainder', 'searchsorted', 'subtract', 'true_divide'
        ]
        v1 = objax.TrainVar(jn.array([1., 2., 3.], dtype=jn.float32))
        v2 = objax.TrainVar(jn.array([10., -20., 30.], dtype=jn.float32))
        for name in API_LIST:
            fn = jn.__dict__[name]
            expected = fn(v1.value, v2.value)
            actual = fn(v1, v2)
            np.testing.assert_allclose(expected, actual)

    def test_jax_duck_typing_jax_api_one_int_arg(self):
        API_LIST = ['bincount', 'bitwise_not', 'invert', 'packbits']
        v = objax.TrainVar(jn.array([1, 2, 3], dtype=jn.int32))
        for name in API_LIST:
            fn = jn.__dict__[name]
            expected = fn(v.value)
            actual = fn(v)
            np.testing.assert_allclose(expected, actual)

    def test_jax_duck_typing_jax_api_two_int_args(self):
        API_LIST = ['bitwise_and', 'bitwise_or', 'bitwise_xor', 'gcd', 'lcm', 'ldexp', 'left_shift', 'right_shift']
        v1 = objax.TrainVar(jn.array([1, 2, 3], dtype=jn.int32))
        v2 = objax.TrainVar(jn.array([10, -20, 30], dtype=jn.int32))
        for name in API_LIST:
            fn = jn.__dict__[name]
            expected = fn(v1.value, v2.value)
            actual = fn(v1, v2)
            np.testing.assert_allclose(expected, actual)

    def test_jax_duck_typing_unary_operators(self):
        API_LIST = ['__neg__', '__pos__', '__abs__', '__invert__']
        v = objax.TrainVar(jn.array([1, 2, 3, -4], dtype=jn.int32))
        for name in API_LIST:
            expected = getattr(v.value, name)()
            actual = getattr(v, name)()
            np.testing.assert_allclose(expected, actual)
        # __round__ required floating point input
        vf = objax.TrainVar(jn.array([1.2, 2.4, 3.7, -4.5], dtype=jn.float32))
        np.testing.assert_allclose(vf.value.__round__(), vf.__round__())

    def test_jax_duck_typing_binary_operators(self):
        API_LIST = [
            '__eq__', '__ne__', '__lt__', '__le__', '__gt__', '__ge__', '__add__', '__radd__', '__sub__', '__rsub__',
            '__mul__', '__rmul__', '__div__', '__rdiv__', '__truediv__', '__rtruediv__', '__floordiv__',
            '__rfloordiv__', '__divmod__', '__rdivmod__', '__mod__', '__rmod__', '__pow__', '__rpow__',
            '__and__', '__rand__', '__or__', '__ror__', '__xor__', '__rxor__', '__lshift__', '__rlshift__',
            '__rshift__', '__rrshift__', '__matmul__', '__rmatmul__',
        ]
        v1 = objax.TrainVar(jn.array([1, 2, 3, -4], dtype=jn.int32))
        v2 = objax.TrainVar(jn.array([1, -2, 5, -4], dtype=jn.int32))
        for name in API_LIST:
            expected = getattr(v1.value, name)(v2.value)
            actual = getattr(v1, name)(v2)
            np.testing.assert_allclose(expected, actual)

    def test_jax_duck_typing_jax_array_methods(self):
        API_LIST = [
            'all', 'any', 'argmax', 'argmin', 'argsort', 'conj', 'conjugate', 'cumprod', 'cumsum', 'diagonal',
            'flatten', 'max', 'mean', 'min', 'nonzero', 'prod', 'ptp', 'ravel', 'round', 'sort', 'squeeze',
            'std', 'sum', 'trace', 'transpose', 'var'
        ]
        v = objax.TrainVar(jn.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=jn.float32))
        for name in API_LIST:
            expected = getattr(v.value, name)()
            actual = getattr(v, name)()
            np.testing.assert_allclose(expected, actual)
        # other methods
        np.testing.assert_allclose(v.value.T, v.T)
        np.testing.assert_allclose(v.value.real, v.real)
        np.testing.assert_allclose(v.value.imag, v.imag)
        np.testing.assert_allclose(v.value.clip(0., 4.), v.clip(0., 4.))
        np.testing.assert_allclose(v.value.dot(v.value), v.dot(v))
        np.testing.assert_allclose(v.value.repeat(3), v.repeat(3))
        np.testing.assert_allclose(v.value.swapaxes(0, 1), v.swapaxes(0, 1))
        np.testing.assert_allclose(v.value.take(np.array([1, 2, 3])), v.take(np.array([1, 2, 3])))
        np.testing.assert_allclose(v.value.tile([1, 2]), v.tile([1, 2]))
        np.testing.assert_allclose(v.value.reshape([-1]), v.reshape([-1]))

    def test_jax_duck_typing_get_item(self):
        v = objax.TrainVar(jn.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=jn.float32))
        np.testing.assert_allclose(v.value[0, 1], v[0, 1])
        np.testing.assert_allclose(v.value[1, :], v[1, :])


if __name__ == '__main__':
    unittest.main()
