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

"""Unitests for automatic variable tracing."""

import unittest

import numpy as np
import jax.numpy as jn
import objax
from objax.zoo.dnnet import DNNet


global_w = objax.TrainVar(jn.zeros(5))
global_b = objax.TrainVar(jn.zeros(1))

global_m = objax.nn.Sequential([objax.nn.Conv2D(2, 4, 3), objax.nn.BatchNorm2D(4)])


class TestTracing(unittest.TestCase):
    """Unit tests for variable tracing using."""

    def test_function_global_vars(self):
        def loss(x, y):
            pred = jn.dot(x, global_w.value) + global_b.value
            return 0.5 * ((y - pred) ** 2).mean()

        vc = objax.util.find_used_variables(loss)
        self.assertDictEqual(vc, {'global_w': global_w, 'global_b': global_b})

    def test_function_global_module(self):
        def loss(x):
            return jn.sum(global_m(x, training=True))

        vc = objax.util.find_used_variables(loss)
        self.assertDictEqual(vc, global_m.vars(scope='global_m.'))

    def test_function_closure_vars(self):
        w = objax.TrainVar(jn.zeros(5))
        b = objax.TrainVar(jn.zeros(1))

        def loss(x, y):
            pred = jn.dot(x, w.value) + b.value
            return 0.5 * ((y - pred) ** 2).mean()

        vc = objax.util.find_used_variables(loss)
        self.assertDictEqual(vc, {'w': w, 'b': b})

    def test_function_closure_module(self):
        m = objax.nn.Sequential([objax.nn.Conv2D(1, 2, 3), objax.nn.BatchNorm2D(2)])

        def loss(x):
            return jn.sum(m(x, training=True))

        vc = objax.util.find_used_variables(loss)
        self.assertDictEqual(vc, m.vars(scope='m.'))

    def test_lambda_with_closure_vars(self):
        w = objax.TrainVar(jn.zeros(5))
        b = objax.TrainVar(jn.zeros(1))

        loss = lambda x, y: 0.5 * ((y - jn.dot(x, w.value) + b.value) ** 2).mean()

        vc = objax.util.find_used_variables(loss)
        self.assertDictEqual(vc, {'w': w, 'b': b})

    def test_multiline_lambda_with_closure_vars(self):
        w = objax.TrainVar(jn.zeros(5))
        b = objax.TrainVar(jn.zeros(1))

        loss = lambda x, y: (
            0.5 * ((y - jn.dot(x, w.value) + b.value) ** 2).mean()
        )

        vc = objax.util.find_used_variables(loss)
        self.assertDictEqual(vc, {'w': w, 'b': b})

    def test_closure_overrides_global_vars(self):
        # Make sure that global variables are what we expect them to be
        np.testing.assert_allclose(global_w.value, np.zeros(5))
        np.testing.assert_allclose(global_b.value, np.zeros(1))

        def _do_test():
            # define local variable with the same name as existing global
            global_w = objax.TrainVar(jn.ones(10))

            # verify that global_w and global_b are what we expect them to be
            np.testing.assert_allclose(global_w.value, np.ones(10))
            np.testing.assert_allclose(global_b.value, np.zeros(1))

            # loss function which mixes closure vars, global vars and closure var hides global var
            def loss(x, y):
                pred = jn.dot(x, global_w.value) + global_b.value
                return 0.5 * ((y - pred) ** 2).mean()

            vc = objax.util.find_used_variables(loss)
            self.assertDictEqual(vc, {'global_w': global_w, 'global_b': global_b})

        _do_test()

        # Make sure that global variables didn't change, in other words
        # that _do_test operated on local variables
        np.testing.assert_allclose(global_w.value, np.zeros(5))
        np.testing.assert_allclose(global_b.value, np.zeros(1))

    def test_typical_training_loop(self):
        # Define model and optimizer
        model = DNNet((32, 10), objax.functional.leaky_relu)
        opt = objax.optimizer.Momentum(model.vars(), nesterov=True)

        # Predict op
        predict_op = lambda x: objax.functional.softmax(model(x, training=False))

        self.assertDictEqual(objax.util.find_used_variables(predict_op),
                             model.vars(scope='model.'))

        # Loss function
        def loss(x, label):
            logit = model(x, training=True)
            xe_loss = objax.functional.loss.cross_entropy_logits_sparse(logit, label).mean()
            return xe_loss

        self.assertDictEqual(objax.util.find_used_variables(loss),
                             model.vars(scope='model.'))

        # Gradients and loss function
        loss_gv = objax.GradValues(loss, objax.util.find_used_variables(loss))

        def train_op(x, y, learning_rate):
            grads, loss = loss_gv(x, y)
            opt(learning_rate, grads)
            return loss

        self.assertDictEqual(objax.util.find_used_variables(train_op),
                             {**model.vars(scope='loss_gv.model.'), **opt.vars(scope='opt.')})

    def test_lambda_inside_function(self):
        m = objax.nn.Sequential([objax.nn.Conv2D(1, 2, 3), objax.nn.BatchNorm2D(2)])

        def loss(x):
            get_logits = lambda inp: m(inp, training=True)
            return jn.sum(get_logits(x))

        vc = objax.util.find_used_variables(loss)
        self.assertDictEqual(vc, m.vars(scope='m.'))


if __name__ == '__main__':
    unittest.main()
