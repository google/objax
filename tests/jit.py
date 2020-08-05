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

import objax


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


if __name__ == '__main__':
    unittest.main()
