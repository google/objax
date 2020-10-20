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

import numpy as np
import tensorflow as tf

import objax
from objax.zoo.resnet_v2 import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNet200
from objax.zoo.resnet_v2 import load_pretrained_weights_from_keras


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


class TestResNetV2Pretrained(unittest.TestCase):

    def check_compatibility(self, model_keras, model_objax):
        data = np.random.uniform(size=(2, 3, 224, 224))
        # training=False
        output_keras = model_keras(data.transpose((0, 2, 3, 1)), training=False).numpy()
        if output_keras.ndim == 4:
            output_keras = output_keras.transpose((0, 3, 1, 2))
        output_objax = model_objax(data, training=False)
        sq_diff = (output_objax - output_keras) ** 2
        self.assertAlmostEqual(sq_diff.mean(), 0, delta=1.001e-06)
        # training=True
        output_keras = model_keras(data.transpose((0, 2, 3, 1)), training=True).numpy()
        if output_keras.ndim == 4:
            output_keras = output_keras.transpose((0, 3, 1, 2))
        output_objax = model_objax(data, training=True)
        sq_diff = (output_objax - output_keras) ** 2
        self.assertAlmostEqual(sq_diff.mean(), 0, delta=1.001e-06)

    def check_output_shape(self, model, num_classes):
        data = np.random.uniform(size=(2, 3, 224, 224))
        output = model(data, training=False)
        self.assertEqual(output.shape, (2, num_classes))

    @parameterized.expand([
        ("ResNet50V2", "ResNet50"),
        ("ResNet101V2", "ResNet101"),
        ("ResNet152V2", "ResNet152"),
    ])
    def test_resnet_include_top(self, keras_name, objax_name):
        model_keras = tf.keras.applications.__dict__[keras_name](include_top=True,
                                                                 weights='imagenet',
                                                                 classes=1000,
                                                                 classifier_activation='linear')
        model_objax = load_pretrained_weights_from_keras(objax_name, include_top=True, num_classes=1000)
        self.check_compatibility(model_keras, model_objax)
        self.check_output_shape(model_objax, 1000)

    @parameterized.expand([
        ("ResNet50V2", "ResNet50"),
        ("ResNet101V2", "ResNet101"),
        ("ResNet152V2", "ResNet152"),
    ])
    def test_resnet_without_top(self, keras_name, objax_name):
        for arch in ['ResNet50', 'ResNet101', 'ResNet152']:
            model_keras = tf.keras.applications.__dict__[keras_name](include_top=False,
                                                                     weights='imagenet',
                                                                     pooling=None)
            model_objax = load_pretrained_weights_from_keras(objax_name, include_top=False, num_classes=10)
            self.check_compatibility(model_keras, model_objax[:-2])
            self.check_output_shape(model_objax, 10)


if __name__ == '__main__':
    unittest.main()
