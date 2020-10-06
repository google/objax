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

"""Unittests for optimizers."""

import math
import unittest

import jax.numpy as jn
import numpy as np

import objax
from objax.variable import TrainVar, VarCollection


class TestOptimizers(unittest.TestCase):
    def __init__(self, methodname):
        """Initialize the test class."""
        super().__init__(methodname)
        self.num_steps = 100
        self.lrs = {'square_adam': 0.15,
                    'rastrigin_adam': 0.7,
                    'logistic_momentum': 1.0,
                    'square_momentum': 0.01,
                    'logistic_momentum_override': 5.0,
                    'square_momentum_override': 0.3,
                    'logistic_sgd': 20.0,
                    'square_sgd': 0.5
                    }
        self.tolerances = {'square_adam': 1e-3,
                           'rastrigin_adam': 1e-3,
                           'logistic_momentum': 1e-10,
                           'square_momentum': 1e-3,
                           'logistic_momentum_override': 1e-3,
                           'square_momentum_override': 1e-3,
                           'logistic_sgd': 1e-10,
                           'square_sgd': 1e-3,
                           }
        self.override_momentums = {'logistic_momentum_override': 0.75,
                                   'square_momentum_override': 0.5,
                                   }

    def _get_optimizer(self, model_vars: VarCollection, optimizer: str):
        if optimizer == 'momentum':
            opt = objax.Jit(objax.optimizer.Momentum(model_vars, momentum=0.9))
        elif optimizer == 'adam':
            opt = objax.Jit(objax.optimizer.Adam(model_vars))
        elif optimizer == 'sgd':
            opt = objax.Jit(objax.optimizer.SGD(model_vars))
        else:
            raise ValueError
        return opt

    def _get_loss(self, loss_name: str):
        if loss_name == 'logistic':
            x = TrainVar(jn.zeros(2))
            model_vars = VarCollection({'x': x})

            def loss():
                return jn.log(jn.exp(-jn.sum(x.value)) + 1)

            return model_vars, loss
        if loss_name == 'square':
            # loss = x*x + y*y.
            x = TrainVar(jn.ones(2))
            y = TrainVar(jn.ones(3))
            model_vars = VarCollection({'x': x, 'y': y})

            def loss():
                return jn.dot(x.value, x.value) + jn.dot(y.value, y.value)

            return model_vars, loss
        if loss_name == 'rastrigin':
            d = 2
            x = TrainVar(jn.ones(d))
            model_vars = VarCollection({'x': x})

            def loss():
                return 10 * d + jn.dot(x.value, x.value) - 10 * jn.sum(jn.cos(2 * math.pi * x.value))

            return model_vars, loss
        raise ValueError

    def _check_run(self, gv, opt, loss, lr, num_steps, tolerance, momentum):
        """Run opt for num_steps times and check if the final loss is small."""
        for i in range(num_steps):
            g, v = gv()
            if momentum:
                opt(lr, g, momentum)
            else:
                opt(lr, g)
        self.assertLess(loss(), tolerance)

    def _test_loss_opt(self, loss_name: str, opt_name: str, override: bool = False):
        """Given loss and optimizer name, get definitions and run test."""
        model_vars, loss = self._get_loss(loss_name)
        gv = objax.GradValues(loss, model_vars)
        opt = self._get_optimizer(model_vars, opt_name)
        test_name = '{}_{}'.format(loss_name, opt_name)
        test_name = test_name + '_override' if override else test_name
        lr = self.lrs[test_name]
        tolerance = self.tolerances[test_name]
        momentum = self.override_momentums[test_name] if override and test_name in self.override_momentums else None
        self._check_run(gv, opt, loss, lr, self.num_steps, tolerance, momentum)
        return model_vars, loss

    def test_square_adam(self):
        """Test square loss for Adam optimizer."""
        model_vars, loss = self._test_loss_opt('square', 'adam')

    def test_rastrigin_adam(self):
        """Test rastrigin loss for Adam optimizer."""
        model_vars, loss = self._test_loss_opt('rastrigin', 'adam')

    def test_logistic_momentum(self):
        """Test logistic loss for momentum optimizer."""
        model_vars, loss = self._test_loss_opt('logistic', 'momentum')

    def test_square_momentum(self):
        """Test square loss for momentum optimizer."""
        model_vars, loss = self._test_loss_opt('square', 'momentum')

    def test_logistic_momentum_override(self):
        """Test logistic loss for momentum optimizer."""
        model_vars, loss = self._test_loss_opt('logistic', 'momentum', True)

    def test_square_momentum_override(self):
        """Test logistic loss for momentum optimizer."""
        model_vars, loss = self._test_loss_opt('square', 'momentum', True)

    def test_logistic_sgd(self):
        """Test logistic loss for sgd optimizer."""
        model_vars, loss = self._test_loss_opt('logistic', 'sgd')

    def test_square_sgd(self):
        """Test square loss for sgd optimizer."""
        model_vars, loss = self._test_loss_opt('square', 'sgd')


class TestEma(unittest.TestCase):
    def test_ema_no_debias(self):
        orig_value_expect = np.array([100.0, -1.0])
        for m in [i / 10.0 for i in range(1, 10)]:
            x = objax.ModuleList([objax.TrainVar(jn.array(orig_value_expect)),
                                  objax.TrainVar(jn.array(orig_value_expect[::-1]))])
            ema = objax.optimizer.ExponentialMovingAverage(x.vars(), momentum=m, debias=False)
            ema()
            ema_value_expect = (1 - m) * orig_value_expect

            def get_tensors():
                return x.vars().tensors()

            get_tensors_ema = ema.replace_vars(get_tensors)
            ema_value = get_tensors_ema()
            orig_value = get_tensors()

            np.testing.assert_allclose(ema_value[0], ema_value_expect, rtol=1e-6)
            np.testing.assert_allclose(ema_value[1], ema_value_expect[::-1], rtol=1e-6)
            np.testing.assert_allclose(orig_value[0], orig_value_expect, rtol=1e-6)
            np.testing.assert_allclose(orig_value[1], orig_value_expect[::-1], rtol=1e-6)

    def test_ema(self):
        eps = 1e-6
        orig_value_expect = np.array([100.0, -1.0])
        for m in [i / 10.0 for i in range(1, 10)]:
            x = objax.ModuleList([objax.TrainVar(jn.array(orig_value_expect))])
            ema = objax.optimizer.ExponentialMovingAverage(x.vars(), momentum=m, debias=True, eps=eps)
            ema()
            ema_value_expect = (1 - m) * orig_value_expect / (1 - (1 - eps) * m)

            def get_tensors():
                return x.vars().tensors()

            get_tensors_ema = ema.replace_vars(get_tensors)
            ema_value = get_tensors_ema()
            orig_value = get_tensors()

            np.testing.assert_allclose(ema_value[0], ema_value_expect, rtol=1e-6)
            np.testing.assert_allclose(orig_value[0], orig_value_expect, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
