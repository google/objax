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

        a, b = objax.functional.scan(cell, jn.zeros((2,)), jn.zeros((3,)))
        self.assertEqual(a.shape, (2,))
        self.assertEqual(b.shape, (3, 2))
