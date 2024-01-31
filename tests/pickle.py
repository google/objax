# Copyright 2021 Google LLC
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

"""Unittests for pickling."""

import unittest

import jax.numpy as jn

import objax

import pickle

class TestPickle(unittest.TestCase):
    def test_on_linear(self):
        """
        Create a Linear module, try to pickle, try to unpickle.
        """
        lin = objax.nn.Linear(2,3)
        pickled = pickle.dumps(lin)
        lin_ = pickle.loads(pickled)
        self.assertTrue(jn.all(lin.w == lin_.w))

if __name__ == '__main__':
    unittest.main()
