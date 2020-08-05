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

from parameterized import parameterized

import objax
from objax.zoo.resnet_v2 import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNet200


class TestResNetV2(unittest.TestCase):

    @parameterized.expand([
        ("ResNet18", ResNet18),
        ("ResNet34", ResNet34),
        ("ResNet50", ResNet50),
        ("ResNet101", ResNet101),
        ("ResNet152", ResNet152),
        ("ResNet200", ResNet200),
    ])
    def test_resnet(self, name, resnet_cls):
        x = objax.random.normal((4, 3, 128, 128))
        model = resnet_cls(in_channels=3, num_classes=10)
        # run in eval mode
        y_eval = model(x, training=False)
        self.assertEqual(y_eval.shape, (4, 10))
        # run in train mode
        y_eval = model(x, training=True)
        self.assertEqual(y_eval.shape, (4, 10))


if __name__ == '__main__':
    unittest.main()
