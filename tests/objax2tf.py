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

"""Unittests for Objax2Tf converter."""

import shutil
import tempfile
import unittest

import numpy as np
import objax
from objax.zoo.wide_resnet import WideResNet
import tensorflow as tf


BATCH_SIZE = 4
NCHANNELS = 3
NCLASSES = 10
IMAGE_SIZE = 32


class TestObjax2Tf(unittest.TestCase):

    def verify_converted_predict_op(self, objax_op, tf_op, shape):
        x1 = np.random.normal(size=shape)
        x2 = np.random.normal(size=shape)
        # due to differences in op implementations, there might be small numerical
        # differences between TF and Objax, thus comparing up to 1e-4 relative tolerance
        np.testing.assert_allclose(objax_op(x1), tf_op(tf.convert_to_tensor(x1, dtype=tf.float32)), rtol=1e-4)
        np.testing.assert_allclose(objax_op(x2), tf_op(tf.convert_to_tensor(x2, dtype=tf.float32)), rtol=1e-4)

    # NOTE: Objax2Tf tests are temporary disabled until the release of TF 2.8

    def disabled_test_convert_wrn(self):
        # Make a model
        model = WideResNet(NCHANNELS, NCLASSES, depth=4, width=1)
        # Prediction op without JIT
        predict_op = objax.nn.Sequential([objax.ForceArgs(model, training=False), objax.functional.softmax])
        predict_tf = objax.util.Objax2Tf(predict_op)
        # Compare results
        self.verify_converted_predict_op(predict_op, predict_tf,
                                         shape=(BATCH_SIZE, NCHANNELS, IMAGE_SIZE, IMAGE_SIZE))
        # Predict op with JIT
        predict_op_jit = objax.Jit(predict_op)
        predict_tf_jit = objax.util.Objax2Tf(predict_op_jit)
        # Compare results
        self.verify_converted_predict_op(predict_op_jit, predict_tf_jit,
                                         shape=(BATCH_SIZE, NCHANNELS, IMAGE_SIZE, IMAGE_SIZE))

    def disabled_test_savedmodel_wrn(self):
        model_dir = tempfile.mkdtemp()
        # Make a model and convert it to TF
        model = WideResNet(NCHANNELS, NCLASSES, depth=4, width=1)
        predict_op = objax.Jit(objax.nn.Sequential([objax.ForceArgs(model, training=False), objax.functional.softmax]))
        predict_tf = objax.util.Objax2Tf(predict_op)
        # Save model
        input_shape = (BATCH_SIZE, NCHANNELS, IMAGE_SIZE, IMAGE_SIZE)
        tf.saved_model.save(
            predict_tf,
            model_dir,
            signatures=predict_tf.__call__.get_concrete_function(tf.TensorSpec(input_shape, tf.float32)))
        # Load model
        loaded_tf_model = tf.saved_model.load(model_dir)
        loaded_predict_tf_op = loaded_tf_model.signatures['serving_default']
        self.verify_converted_predict_op(predict_op,
                                         lambda x: loaded_predict_tf_op(x)['output_0'],
                                         shape=input_shape)
        self.verify_converted_predict_op(predict_op,
                                         lambda x: loaded_tf_model(x),
                                         shape=input_shape)
        # Cleanup
        shutil.rmtree(model_dir)


if __name__ == '__main__':
    unittest.main()
