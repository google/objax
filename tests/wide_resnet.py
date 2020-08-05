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

"""Unittests for Resnet v2."""

import unittest

import objax
from objax.zoo.wide_resnet import WideResNet, WideResNetGeneral


class TestWideResNetGeneral(unittest.TestCase):

    def test_wide_resnet_general(self):
        x = objax.random.normal((4, 3, 128, 128))
        model = WideResNetGeneral(nin=3, nclass=10, blocks_per_group=[4, 4, 4, 4], width=2)
        # run in eval mode
        y_eval = model(x, training=False)
        self.assertEqual(y_eval.shape, (4, 10))
        # run in train mode
        y_eval = model(x, training=True)
        self.assertEqual(y_eval.shape, (4, 10))

    def test_wide_resnet(self):
        x = objax.random.normal((4, 3, 32, 32))
        model = WideResNet(nin=3, nclass=10, depth=28, width=4)
        # run in eval mode
        y_eval = model(x, training=False)
        self.assertEqual(y_eval.shape, (4, 10))
        # run in train mode
        y_eval = model(x, training=True)
        self.assertEqual(y_eval.shape, (4, 10))


if __name__ == '__main__':
    unittest.main()
