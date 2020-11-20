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

"""Unittests for scan method."""

import unittest

import jax.numpy as jn

import objax


class TestScan(unittest.TestCase):
    def test_scan(self):
        def cell(carry, x):
            return jn.array([2]) * carry * x, jn.array([2]) * carry * x

        carry = jn.array([8., 8.])
        output = jn.array([[2., 2.], [4., 4.], [8., 8.]])
        test_carry, test_output = objax.functional.scan(cell, jn.ones((2,)), jn.ones((3,)))
        self.assertTrue(jn.array_equal(carry, test_carry))
        self.assertTrue(jn.array_equal(output, test_output))
