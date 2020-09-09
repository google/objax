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

"""Unittests for Vectorize Layer."""

import unittest

import jax.numpy as jn

import objax


class TestVectorize(unittest.TestCase):

    def test_vectorize_module_one_arg(self):
        """Vectorize module with a single argument."""
        f = objax.nn.Linear(3, 4)
        fv = objax.Vectorize(f)
        x = objax.random.normal((96, 3))
        y = f(x)
        yv = fv(x)
        self.assertTrue(jn.array_equal(y, yv))

    def test_vectorize_module_one_arg_transposed_axis(self):
        """Vectorize module with a single argument transposed axis."""
        f = objax.nn.Linear(3, 4)
        fv = objax.Vectorize(f, batch_axis=(1,))
        x = objax.random.normal((96, 3))
        y = f(x)
        yv = fv(x.T)
        self.assertTrue(jn.array_equal(y, yv))

    def test_vectorize_module_two_args(self):
        """Vectorize module with a two arguments."""
        f1 = objax.nn.Linear(3, 4)
        f2 = objax.nn.Linear(5, 3)
        f = lambda x, y: jn.concatenate([f1(x), f2(y)], axis=1)
        fv = objax.Vectorize(lambda x, y: jn.concatenate([f1(x), f2(y)], axis=0), f1.vars('f1') + f2.vars('f2'),
                             batch_axis=(0, 0))
        x1 = objax.random.normal((96, 3))
        x2 = objax.random.normal((96, 5))
        y = f(x1, x2)
        yv = fv(x1, x2)
        self.assertTrue(jn.array_equal(y, yv))

    def test_vectorize_module_two_args_mixed_axis(self):
        """Vectorize module with a two arguments with mixed batch axis."""
        f1 = objax.nn.Linear(3, 4)
        f2 = objax.nn.Linear(5, 3)
        f = lambda x, y: jn.concatenate([f1(x), f2(y)], axis=1)
        fv = objax.Vectorize(lambda x, y: jn.concatenate([f1(x), f2(y)], axis=0), f1.vars('f1') + f2.vars('f2'),
                             batch_axis=(0, 1))
        x1 = objax.random.normal((96, 3))
        x2 = objax.random.normal((96, 5))
        y = f(x1, x2)
        yv = fv(x1, x2.T)
        self.assertTrue(jn.array_equal(y, yv))

    def test_vectorize_module_one_arg_one_static(self):
        """Vectorize module with a broadcast argument and a static one."""
        f = objax.nn.Linear(3, 4)
        c = objax.random.normal([4])
        fv = objax.Vectorize(lambda x, y: f(x) + y, f.vars(), batch_axis=(0, None))
        x = objax.random.normal((96, 3))
        y = f(x) + c
        yv = fv(x, c)
        self.assertTrue(jn.array_equal(y, yv))

    def test_vectorize_module_one_arg_one_static_positional_syntax(self):
        """Vectorize module with a broadcast argument and a static one using positional syntax."""
        f = objax.nn.Linear(3, 4)
        c = objax.random.normal([4])
        fv = objax.Vectorize(lambda *a: f(a[0]) + a[1], f.vars(), batch_axis=(0, None))
        x = objax.random.normal((96, 3))
        y = f(x) + c
        yv = fv(x, c)
        self.assertTrue(jn.array_equal(y, yv))

    def test_vectorize_module_one_arg_one_static_missing_batched(self):
        """Vectorize module with a broadcast argument and a static one with incomplete batch argument."""
        f = objax.nn.Linear(3, 4)
        with self.assertRaises(AssertionError):
            _ = objax.Vectorize(lambda x, y: f(x) + y, f.vars(), batch_axis=(0,))

    def test_vectorize_module_one_arg_one_static_missing_batched_call(self):
        """Vectorize module with a broadcast argument and a static one with incomplete batch argument.
        Catch exception during call for variadic functions."""
        f = objax.nn.Linear(3, 4)
        c = objax.random.normal([4])
        fv = objax.Vectorize(lambda *a: f(a[0]) + a[1], f.vars(), batch_axis=(0,))
        x = objax.random.normal((96, 3))
        with self.assertRaises(AssertionError):
            _ = fv(x, c)

    def test_vectorize_random_function(self):
        class RandomReverse(objax.Module):
            def __init__(self):
                self.keygen = objax.random.Generator()

            def __call__(self, x):
                r = objax.random.randint([], low=0, high=2, generator=self.keygen)
                return x + r * (x[::-1] - x), r

        random_reverse = RandomReverse()
        vector_reverse = objax.Vectorize(random_reverse)
        x = jn.arange(20).reshape((10, 2))
        y, r = vector_reverse(x)
        self.assertEqual(r.tolist(), [1, 1, 1, 1, 1, 0, 0, 1, 0, 0])
        self.assertEqual(y.tolist(), [[1, 0], [3, 2], [5, 4], [7, 6], [9, 8],
                                      [10, 11], [12, 13], [15, 14], [16, 17], [18, 19]])

    def test_vectorize_random_function_reseed(self):
        class RandomReverse(objax.Module):
            def __init__(self):
                self.keygen = objax.random.Generator(1337)

            def __call__(self, x):
                r = objax.random.randint([], 0, 2, generator=self.keygen)
                return x + r * (x[::-1] - x), r

        random_reverse = RandomReverse()
        vector_reverse = objax.Vectorize(random_reverse)
        x = jn.arange(20).reshape((10, 2))
        random_reverse.keygen.seed(0)
        y, r = vector_reverse(x)
        self.assertEqual(r.tolist(), [1, 1, 1, 1, 1, 0, 0, 1, 0, 0])
        self.assertEqual(y.tolist(), [[1, 0], [3, 2], [5, 4], [7, 6], [9, 8],
                                      [10, 11], [12, 13], [15, 14], [16, 17], [18, 19]])


if __name__ == '__main__':
    unittest.main()
