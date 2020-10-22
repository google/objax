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

"""Unittests for objax.random."""

import unittest

import numpy as np
import scipy.stats

import objax


class TestRandom(unittest.TestCase):

    def helper_test_randint(self, shape, low, high):
        """Helper function to test objax.random.randint."""
        value = objax.random.randint(shape, low, high)
        self.assertEqual(value.shape, shape)
        self.assertTrue(np.all(value >= low))
        self.assertTrue(np.all(value < high))

    def test_randint(self):
        """Test for objax.random.randint."""
        self.helper_test_randint(shape=(3, 4), low=1, high=10)
        self.helper_test_randint(shape=(5,), low=0, high=5)
        self.helper_test_randint(shape=(), low=-5, high=5)

    def helper_test_normal(self, shape, stddev):
        """Helper function to test objax.random.normal."""
        value = objax.random.normal(shape, stddev=stddev)
        self.assertEqual(value.shape, shape)

    def test_normal(self):
        """Test for objax.random.normal."""
        self.helper_test_normal(shape=(4, 2, 3), stddev=1.0)
        self.helper_test_normal(shape=(2, 3), stddev=2.0)
        self.helper_test_normal(shape=(5,), stddev=2.0)
        self.helper_test_normal(shape=(), stddev=10.0)
        value = np.array(objax.random.normal((1000, 100)))
        self.assertAlmostEqual(value.mean(), 0, delta=0.01)
        self.assertAlmostEqual(value.std(), 1, delta=0.01)
        value = np.array(objax.random.normal((1000, 100), mean=0, stddev=2))
        self.assertAlmostEqual(value.mean(), 0, delta=0.02)
        self.assertAlmostEqual(value.std(), 2, delta=0.01)
        value = np.array(objax.random.normal((1000, 100), mean=1, stddev=1.5))
        self.assertAlmostEqual(value.mean(), 1, delta=0.015)
        self.assertAlmostEqual(value.std(), 1.5, delta=0.01)

    def helper_test_truncated_normal(self, shape, stddev, bound):
        """Helper function to test objax.random.truncated_normal."""
        value = objax.random.truncated_normal(shape, stddev=stddev, lower=-bound, upper=bound)
        self.assertEqual(value.shape, shape)
        self.assertTrue(np.all(value >= -bound * stddev))
        self.assertTrue(np.all(value <= bound * stddev))

    def test_truncated_normal(self):
        """Test for objax.random.truncated_normal."""
        self.helper_test_truncated_normal(shape=(5, 7), stddev=1.0, bound=2.0)
        self.helper_test_truncated_normal(shape=(4,), stddev=2.0, bound=4.0)
        self.helper_test_truncated_normal(shape=(), stddev=1.0, bound=4.0)
        value = np.array(objax.random.truncated_normal((1000, 100)))
        truncated_std = scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1)
        self.assertAlmostEqual(value.mean(), 0, delta=0.01)
        self.assertAlmostEqual(value.std(), truncated_std, delta=0.01)
        self.assertAlmostEqual(value.min(), -1.9, delta=0.1)
        self.assertAlmostEqual(value.max(), 1.9, delta=0.1)
        value = np.array(objax.random.truncated_normal((1000, 100), stddev=2, lower=-3, upper=3))
        truncated_std = scipy.stats.truncnorm.std(a=-3, b=3, loc=0., scale=2)
        self.assertAlmostEqual(value.mean(), 0, delta=0.01)
        self.assertAlmostEqual(value.std(), truncated_std, delta=0.01)
        self.assertAlmostEqual(value.min(), -5.9, delta=0.1)
        self.assertAlmostEqual(value.max(), 5.9, delta=0.1)
        value = np.array(objax.random.truncated_normal((1000, 100), stddev=1.5))
        truncated_std = scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.5)
        self.assertAlmostEqual(value.mean(), 0, delta=0.01)
        self.assertAlmostEqual(value.std(), truncated_std, delta=0.01)
        self.assertAlmostEqual(value.min(), -2.9, delta=0.1)
        self.assertAlmostEqual(value.max(), 2.9, delta=0.1)

    def helper_test_uniform(self, shape):
        """Helper function to test objax.random.uniform."""
        value = objax.random.uniform(shape)
        self.assertEqual(value.shape, shape)
        self.assertTrue(np.all(value >= 0.0))
        self.assertTrue(np.all(value < 1.0))

    def test_uniform(self):
        """Test for objax.random.uniform."""
        self.helper_test_uniform(shape=(4, 3))
        self.helper_test_uniform(shape=(5,))
        self.helper_test_uniform(shape=())

    def test_generator(self):
        """Test for objax.random.Generator."""
        g1 = objax.random.Generator(0)
        g2 = objax.random.Generator(0)
        g3 = objax.random.Generator(1)
        value1 = objax.random.randint((3, 4), low=0, high=65536, generator=g1)
        value2 = objax.random.randint((3, 4), low=0, high=65536, generator=g2)
        value3 = objax.random.randint((3, 4), low=0, high=65536, generator=g3)
        self.assertTrue(np.all(value1 == value2))
        self.assertFalse(np.all(value1 == value3))

        g4 = objax.random.Generator(123)
        value = [objax.random.randint(shape=(1,), low=0, high=65536, generator=g4) for _ in range(2)]
        self.assertNotEqual(value[0], value[1])


if __name__ == '__main__':
    unittest.main()
