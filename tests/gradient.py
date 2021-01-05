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
"""Unittests for Grad, GradValues and PrivateGradValues."""

import inspect
import unittest
from typing import Tuple, Dict, List

import jax
import jax.numpy as jn
import numpy as np
from scipy.stats.distributions import chi2

import objax
from objax.typing import JaxArray
from objax.zoo.dnnet import DNNet


class TestGrad(unittest.TestCase):
    def test_grad_linear(self):
        """Test if gradient has the correct value for linear regression."""
        # Set data
        ndim = 2
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [-10.0, 9.0]])
        labels = np.array([1.0, 2.0, 3.0, 4.0])

        # Set model parameters for linear regression.
        w = objax.TrainVar(jn.zeros(ndim))
        b = objax.TrainVar(jn.zeros(1))

        def loss(x, y):
            pred = jn.dot(x, w.value) + b.value
            return 0.5 * ((y - pred) ** 2).mean()

        grad = objax.Grad(loss, objax.VarCollection({'w': w, 'b': b}))
        g = grad(data, labels)

        self.assertEqual(g[0].shape, tuple([ndim]))
        self.assertEqual(g[1].shape, tuple([1]))

        g_expect_w = -(data * np.tile(labels, (ndim, 1)).transpose()).mean(0)
        g_expect_b = np.array([-labels.mean()])
        np.testing.assert_allclose(g[0], g_expect_w)
        np.testing.assert_allclose(g[1], g_expect_b)

    def test_grad_linear_and_inputs(self):
        """Test if gradient of inputs and variables has the correct values for linear regression."""
        # Set data
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [-10.0, 9.0]])
        labels = np.array([1.0, 2.0, 3.0, 4.0])

        # Set model parameters for linear regression.
        w = objax.TrainVar(jn.array([2, 3], jn.float32))
        b = objax.TrainVar(jn.array([1], jn.float32))

        def loss(x, y):
            pred = jn.dot(x, w.value) + b.value
            return 0.5 * ((y - pred) ** 2).mean()

        expect_gw = [37.25, 69.0]
        expect_gb = [13.75]
        expect_gx = [[4.0, 6.0], [8.5, 12.75], [13.0, 19.5], [2.0, 3.0]]
        expect_gy = [-2.0, -4.25, -6.5, -1.0]

        grad0 = objax.Grad(loss, objax.VarCollection({'w': w, 'b': b}), input_argnums=(0,))
        g = grad0(data, labels)
        self.assertEqual(g[0].tolist(), expect_gx)
        self.assertEqual(g[1].tolist(), expect_gw)
        self.assertEqual(g[2].tolist(), expect_gb)

        grad1 = objax.Grad(loss, objax.VarCollection({'w': w, 'b': b}), input_argnums=(1,))
        g = grad1(data, labels)
        self.assertEqual(g[0].tolist(), expect_gy)
        self.assertEqual(g[1].tolist(), expect_gw)
        self.assertEqual(g[2].tolist(), expect_gb)

        grad01 = objax.Grad(loss, objax.VarCollection({'w': w, 'b': b}), input_argnums=(0, 1))
        g = grad01(data, labels)
        self.assertEqual(g[0].tolist(), expect_gx)
        self.assertEqual(g[1].tolist(), expect_gy)
        self.assertEqual(g[2].tolist(), expect_gw)
        self.assertEqual(g[3].tolist(), expect_gb)

        grad10 = objax.Grad(loss, objax.VarCollection({'w': w, 'b': b}), input_argnums=(1, 0))
        g = grad10(data, labels)
        self.assertEqual(g[0].tolist(), expect_gy)
        self.assertEqual(g[1].tolist(), expect_gx)
        self.assertEqual(g[2].tolist(), expect_gw)
        self.assertEqual(g[3].tolist(), expect_gb)

        grad10 = objax.Grad(loss, None, input_argnums=(0, 1))
        g = grad10(data, labels)
        self.assertEqual(g[0].tolist(), expect_gx)
        self.assertEqual(g[1].tolist(), expect_gy)

    def test_grad_logistic(self):
        """Test if gradient has the correct value for logistic regression."""
        # Set data
        ndim = 2
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [-10.0, 9.0]])
        labels = np.array([1.0, -1.0, 1.0, -1.0])

        # Set model parameters for linear regression.
        w = objax.TrainVar(jn.ones(ndim))

        def loss(x, y):
            xyw = jn.dot(x * np.tile(y, (ndim, 1)).transpose(), w.value)
            return jn.log(jn.exp(-xyw) + 1).mean(0)

        grad = objax.Grad(loss, objax.VarCollection({'w': w}))
        g = grad(data, labels)

        self.assertEqual(g[0].shape, tuple([ndim]))

        xw = np.dot(data, w.value)
        g_expect_w = -(data * np.tile(labels / (1 + np.exp(labels * xw)), (ndim, 1)).transpose()).mean(0)
        np.testing.assert_allclose(g[0], g_expect_w, atol=1e-7)

    def test_grad_constant(self):
        """Test if constants are preserved."""
        # Set data
        ndim = 1
        data = np.array([[1.0], [3.0], [5.0], [-10.0]])
        labels = np.array([1.0, 2.0, 3.0, 4.0])

        # Set model parameters for linear regression.
        w = objax.TrainVar(jn.zeros(ndim))
        b = objax.TrainVar(jn.ones(1))
        m = objax.ModuleList([w, b])

        def loss(x, y):
            pred = jn.dot(x, w.value) + b.value
            return 0.5 * ((y - pred) ** 2).mean()

        # We are supposed to see the gradient change after the value of b (the constant) changes.
        grad = objax.Grad(loss, objax.VarCollection({'w': w}))
        g_old = grad(data, labels)
        b.assign(-b.value)
        g_new = grad(data, labels)
        self.assertNotEqual(g_old[0][0], g_new[0][0])

        # When compile with Jit, we are supposed to see the gradient change after the value of b (the constant) changes.
        grad = objax.Jit(objax.Grad(loss, objax.VarCollection({'w': w})), m.vars())
        g_old = grad(data, labels)
        b.assign(-b.value)
        g_new = grad(data, labels)
        self.assertNotEqual(g_old[0][0], g_new[0][0])

    def test_grad_signature(self):
        def f(x: JaxArray, y) -> Tuple[JaxArray, Dict[str, JaxArray]]:
            return (x + y).mean(), {'x': x, 'y': y}

        def df(x: JaxArray, y) -> List[JaxArray]:
            pass  # Signature of the differential of f

        g = objax.Grad(f, objax.VarCollection())
        self.assertEqual(inspect.signature(g), inspect.signature(df))

    def test_trainvar_assign(self):
        # Set data
        ndim = 2
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [-10.0, 9.0]])
        labels = np.array([1.0, 2.0, 3.0, 4.0])

        # Set model parameters for linear regression.
        w = objax.TrainVar(jn.zeros(ndim))
        b = objax.TrainVar(jn.zeros(1))

        def loss(x, y):
            pred = jn.dot(x, w.value) + b.value
            b.assign(b.value + 1)
            w.assign(w.value - 1)
            return 0.5 * ((y - pred) ** 2).mean()

        grad = objax.Grad(loss, objax.VarCollection({'w': w, 'b': b}))

        def jloss(wb, x, y):
            w, b = wb
            pred = jn.dot(x, w) + b
            return 0.5 * ((y - pred) ** 2).mean()

        jgrad = jax.grad(jloss)

        jg = jgrad([w.value, b.value], data, labels)
        g = grad(data, labels)
        self.assertEqual(g[0].shape, tuple([ndim]))
        self.assertEqual(g[1].shape, tuple([1]))
        np.testing.assert_allclose(g[0], jg[0])
        np.testing.assert_allclose(g[1], jg[1])
        self.assertEqual(w.value.tolist(), [-1., -1.])
        self.assertEqual(b.value.tolist(), [1.])

        jg = jgrad([w.value, b.value], data, labels)
        g = grad(data, labels)
        np.testing.assert_allclose(g[0], jg[0])
        np.testing.assert_allclose(g[1], jg[1])
        self.assertEqual(w.value.tolist(), [-2., -2.])
        self.assertEqual(b.value.tolist(), [2.])

    def test_trainvar_preassign(self):
        # Set data
        ndim = 2
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [-10.0, 9.0]])
        labels = np.array([1.0, 2.0, 3.0, 4.0])

        # Set model parameters for linear regression.
        w = objax.TrainVar(jn.zeros(ndim))
        b = objax.TrainVar(jn.zeros(1))

        def loss(x, y):
            b.assign(b.value + 1)
            w.assign(w.value - 1)
            pred = jn.dot(x, w.value) + b.value
            return 0.5 * ((y - pred) ** 2).mean()

        grad = objax.Grad(loss, objax.VarCollection({'w': w, 'b': b}))

        def jloss(wb, x, y):
            w, b = wb
            pred = jn.dot(x, w) + b
            return 0.5 * ((y - pred) ** 2).mean()

        jgrad = jax.grad(jloss)

        g = grad(data, labels)
        jg = jgrad([w.value, b.value], data, labels)
        self.assertEqual(g[0].shape, tuple([ndim]))
        self.assertEqual(g[1].shape, tuple([1]))
        np.testing.assert_allclose(g[0], jg[0])
        np.testing.assert_allclose(g[1], jg[1])
        self.assertEqual(w.value.tolist(), [-1., -1.])
        self.assertEqual(b.value.tolist(), [1.])

        g = grad(data, labels)
        jg = jgrad([w.value, b.value], data, labels)
        np.testing.assert_allclose(g[0], jg[0])
        np.testing.assert_allclose(g[1], jg[1])
        self.assertEqual(w.value.tolist(), [-2., -2.])
        self.assertEqual(b.value.tolist(), [2.])

    def test_trainvar_jit_assign(self):
        # Set data
        ndim = 2
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [-10.0, 9.0]])
        labels = np.array([1.0, 2.0, 3.0, 4.0])

        # Set model parameters for linear regression.
        w = objax.TrainVar(jn.zeros(ndim))
        b = objax.TrainVar(jn.zeros(1))

        def loss(x, y):
            pred = jn.dot(x, w.value) + b.value
            b.assign(b.value + 1)
            w.assign(w.value - 1)
            return 0.5 * ((y - pred) ** 2).mean()

        grad = objax.Grad(loss, objax.VarCollection({'w': w, 'b': b}))

        def jloss(wb, x, y):
            w, b = wb
            pred = jn.dot(x, w) + b
            return 0.5 * ((y - pred) ** 2).mean()

        def jit_op(x, y):
            g = grad(x, y)
            b.assign(b.value * 2)
            w.assign(w.value * 3)
            return g

        jit_op = objax.Jit(jit_op, objax.VarCollection(dict(b=b, w=w)))
        jgrad = jax.grad(jloss)

        jg = jgrad([w.value, b.value], data, labels)
        g = jit_op(data, labels)
        self.assertEqual(g[0].shape, tuple([ndim]))
        self.assertEqual(g[1].shape, tuple([1]))
        np.testing.assert_allclose(g[0], jg[0])
        np.testing.assert_allclose(g[1], jg[1])
        self.assertEqual(w.value.tolist(), [-3., -3.])
        self.assertEqual(b.value.tolist(), [2.])

        jg = jgrad([w.value, b.value], data, labels)
        g = jit_op(data, labels)
        np.testing.assert_allclose(g[0], jg[0])
        np.testing.assert_allclose(g[1], jg[1])
        self.assertEqual(w.value.tolist(), [-12., -12.])
        self.assertEqual(b.value.tolist(), [6.])


class TestGradValues(unittest.TestCase):
    def test_gradvalues_linear(self):
        """Test if gradient has the correct value for linear regression."""
        # Set data
        ndim = 2
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [-10.0, 9.0]])
        labels = np.array([1.0, 2.0, 3.0, 4.0])

        # Set model parameters for linear regression.
        w = objax.TrainVar(jn.zeros(ndim))
        b = objax.TrainVar(jn.zeros(1))

        def loss(x, y):
            pred = jn.dot(x, w.value) + b.value
            return 0.5 * ((y - pred) ** 2).mean()

        gv = objax.GradValues(loss, objax.VarCollection({'w': w, 'b': b}))
        g, v = gv(data, labels)

        self.assertEqual(g[0].shape, tuple([ndim]))
        self.assertEqual(g[1].shape, tuple([1]))

        g_expect_w = -(data * np.tile(labels, (ndim, 1)).transpose()).mean(0)
        g_expect_b = np.array([-labels.mean()])
        np.testing.assert_allclose(g[0], g_expect_w)
        np.testing.assert_allclose(g[1], g_expect_b)
        np.testing.assert_allclose(v[0], loss(data, labels))

    def test_gradvalues_linear_and_inputs(self):
        """Test if gradient of inputs and variables has the correct values for linear regression."""
        # Set data
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [-10.0, 9.0]])
        labels = np.array([1.0, 2.0, 3.0, 4.0])

        # Set model parameters for linear regression.
        w = objax.TrainVar(jn.array([2, 3], jn.float32))
        b = objax.TrainVar(jn.array([1], jn.float32))

        def loss(x, y):
            pred = jn.dot(x, w.value) + b.value
            return 0.5 * ((y - pred) ** 2).mean()

        expect_loss = loss(data, labels)
        expect_gw = [37.25, 69.0]
        expect_gb = [13.75]
        expect_gx = [[4.0, 6.0], [8.5, 12.75], [13.0, 19.5], [2.0, 3.0]]
        expect_gy = [-2.0, -4.25, -6.5, -1.0]

        gv0 = objax.GradValues(loss, objax.VarCollection({'w': w, 'b': b}), input_argnums=(0,))
        g, v = gv0(data, labels)
        self.assertEqual(v[0], expect_loss)
        self.assertEqual(g[0].tolist(), expect_gx)
        self.assertEqual(g[1].tolist(), expect_gw)
        self.assertEqual(g[2].tolist(), expect_gb)

        gv1 = objax.GradValues(loss, objax.VarCollection({'w': w, 'b': b}), input_argnums=(1,))
        g, v = gv1(data, labels)
        self.assertEqual(v[0], expect_loss)
        self.assertEqual(g[0].tolist(), expect_gy)
        self.assertEqual(g[1].tolist(), expect_gw)
        self.assertEqual(g[2].tolist(), expect_gb)

        gv01 = objax.GradValues(loss, objax.VarCollection({'w': w, 'b': b}), input_argnums=(0, 1))
        g, v = gv01(data, labels)
        self.assertEqual(v[0], expect_loss)
        self.assertEqual(g[0].tolist(), expect_gx)
        self.assertEqual(g[1].tolist(), expect_gy)
        self.assertEqual(g[2].tolist(), expect_gw)
        self.assertEqual(g[3].tolist(), expect_gb)

        gv10 = objax.GradValues(loss, objax.VarCollection({'w': w, 'b': b}), input_argnums=(1, 0))
        g, v = gv10(data, labels)
        self.assertEqual(v[0], expect_loss)
        self.assertEqual(g[0].tolist(), expect_gy)
        self.assertEqual(g[1].tolist(), expect_gx)
        self.assertEqual(g[2].tolist(), expect_gw)
        self.assertEqual(g[3].tolist(), expect_gb)

        gv10 = objax.GradValues(loss, None, input_argnums=(0, 1))
        g, v = gv10(data, labels)
        self.assertEqual(v[0], expect_loss)
        self.assertEqual(g[0].tolist(), expect_gx)
        self.assertEqual(g[1].tolist(), expect_gy)

    def test_gradvalues_logistic(self):
        """Test if gradient has the correct value for logistic regression."""
        # Set data
        ndim = 2
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [-10.0, 9.0]])
        labels = np.array([1.0, -1.0, 1.0, -1.0])

        # Set model parameters for linear regression.
        w = objax.TrainVar(jn.ones(ndim))

        def loss(x, y):
            xyw = jn.dot(x * np.tile(y, (ndim, 1)).transpose(), w.value)
            return jn.log(jn.exp(-xyw) + 1).mean(0)

        gv = objax.GradValues(loss, objax.VarCollection({'w': w}))
        g, v = gv(data, labels)

        self.assertEqual(g[0].shape, tuple([ndim]))

        xw = np.dot(data, w.value)
        g_expect_w = -(data * np.tile(labels / (1 + np.exp(labels * xw)), (ndim, 1)).transpose()).mean(0)
        np.testing.assert_allclose(g[0], g_expect_w, atol=1e-7)
        np.testing.assert_allclose(v[0], loss(data, labels))

    def test_gradvalues_constant(self):
        """Test if constants are preserved."""
        # Set data
        ndim = 1
        data = np.array([[1.0], [3.0], [5.0], [-10.0]])
        labels = np.array([1.0, 2.0, 3.0, 4.0])

        # Set model parameters for linear regression.
        w = objax.TrainVar(jn.zeros(ndim))
        b = objax.TrainVar(jn.ones(1))
        m = objax.ModuleList([w, b])

        def loss(x, y):
            pred = jn.dot(x, w.value) + b.value
            return 0.5 * ((y - pred) ** 2).mean()

        # We are supposed to see the gradient change after the value of b (the constant) changes.
        gv = objax.GradValues(loss, objax.VarCollection({'w': w}))
        g_old, v_old = gv(data, labels)
        b.assign(-b.value)
        g_new, v_new = gv(data, labels)
        self.assertNotEqual(g_old[0][0], g_new[0][0])

        # When compile with Jit, we are supposed to see the gradient change after the value of b (the constant) changes.
        gv = objax.Jit(objax.GradValues(loss, objax.VarCollection({'w': w})), m.vars())
        g_old, v_old = gv(data, labels)
        b.assign(-b.value)
        g_new, v_new = gv(data, labels)
        self.assertNotEqual(g_old[0][0], g_new[0][0])

    def test_gradvalues_signature(self):
        def f(x: JaxArray, y) -> Tuple[JaxArray, Dict[str, JaxArray]]:
            return (x + y).mean(), {'x': x, 'y': y}

        def df(x: JaxArray, y) -> Tuple[List[JaxArray], Tuple[JaxArray, Dict[str, JaxArray]]]:
            pass  # Signature of the (differential of f, f)

        g = objax.GradValues(f, objax.VarCollection())
        self.assertEqual(inspect.signature(g), inspect.signature(df))


class TestPrivateGradValues(unittest.TestCase):
    def __init__(self, methodname):
        """Initialize the test class."""
        super().__init__(methodname)

        self.ntrain = 100
        self.nclass = 5
        self.ndim = 5

        # Generate random data.
        np.random.seed(1234)
        self.data = np.random.rand(self.ntrain, self.ndim) * 10
        self.labels = np.random.randint(self.nclass, size=self.ntrain)
        self.labels = (np.arange(self.nclass) == self.labels[:, None]).astype('f')  # make one-hot

        # Set model, optimizer and loss.
        self.model = DNNet(layer_sizes=[self.ndim, self.nclass], activation=objax.functional.softmax)
        self.model_vars = self.model.vars()

        def loss_function(x, y):
            logit = self.model(x)
            loss = ((y - logit) ** 2).mean(1).mean(0)
            return loss, {'loss': loss}

        self.loss = loss_function

    def test_private_gradvalues_compare_nonpriv(self):
        """Test if PrivateGradValues without clipping / noise is the same as non-private GradValues."""
        l2_norm_clip = 1e10
        noise_multiplier = 0

        for use_norm_accumulation in [True, False]:
            for microbatch in [1, 10, self.ntrain]:
                gv_priv = objax.Jit(objax.privacy.dpsgd.PrivateGradValues(self.loss, self.model_vars,
                                                                          noise_multiplier, l2_norm_clip, microbatch,
                                                                          batch_axis=(0, 0),
                                                                          use_norm_accumulation=use_norm_accumulation))
                gv = objax.GradValues(self.loss, self.model_vars)
                g_priv, v_priv = gv_priv(self.data, self.labels)
                g, v = gv(self.data, self.labels)

                # Check the shape of the gradient.
                self.assertEqual(g_priv[0].shape, tuple([self.nclass]))
                self.assertEqual(g_priv[1].shape, tuple([self.ndim, self.nclass]))

                # Check if the private gradient is similar to the non-private gradient.
                np.testing.assert_allclose(g[0], g_priv[0], atol=1e-7)
                np.testing.assert_allclose(g[1], g_priv[1], atol=1e-7)
                np.testing.assert_allclose(v_priv[0], self.loss(self.data, self.labels)[0], atol=1e-7)

    def test_private_gradvalues_clipping(self):
        """Test if the gradient norm is within l2_norm_clip."""
        noise_multiplier = 0
        acceptable_float_error = 1e-8
        for use_norm_accumulation in [True, False]:
            for microbatch in [1, 10, self.ntrain]:
                for l2_norm_clip in [0, 1e-2, 1e-1, 1.0]:
                    gv_priv = objax.Jit(objax.privacy.dpsgd.PrivateGradValues(
                        self.loss, self.model_vars,
                        noise_multiplier, l2_norm_clip, microbatch,
                        batch_axis=(0, 0),
                        use_norm_accumulation=use_norm_accumulation))
                    g_priv, v_priv = gv_priv(self.data, self.labels)
                    # Get the actual squared norm of the gradient.
                    g_normsquared = sum([np.sum(g ** 2) for g in g_priv])
                    self.assertLessEqual(g_normsquared, l2_norm_clip ** 2 + acceptable_float_error)
                    np.testing.assert_allclose(v_priv[0], self.loss(self.data, self.labels)[0], atol=1e-7)

    def test_private_gradvalues_noise(self):
        """Test if the noise std is around expected."""
        runs = 100
        alpha = 0.0001
        for use_norm_accumulation in [True, False]:
            for microbatch in [1, 10, self.ntrain]:
                for noise_multiplier in [0.1, 10.0]:
                    for l2_norm_clip in [0.01, 0.1]:
                        gv_priv = objax.Jit(objax.privacy.dpsgd.PrivateGradValues(
                            self.loss,
                            self.model_vars,
                            noise_multiplier,
                            l2_norm_clip,
                            microbatch,
                            batch_axis=(0, 0),
                            use_norm_accumulation=use_norm_accumulation))
                        # Repeat the run and collect all gradients.
                        g_privs = []
                        for i in range(runs):
                            g_priv, v_priv = gv_priv(self.data, self.labels)
                            g_privs.append(np.concatenate([g_n.reshape(-1) for g_n in g_priv]))
                            np.testing.assert_allclose(v_priv[0], self.loss(self.data, self.labels)[0], atol=1e-7)
                        g_privs = np.array(g_privs)

                        # Compute empirical std and expected std.
                        std_empirical = np.std(g_privs, axis=0, ddof=1)
                        std_theoretical = l2_norm_clip * noise_multiplier / (self.ntrain // microbatch)

                        # Conduct chi-square test for correct expected standard
                        # deviation.
                        chi2_value = (runs - 1) * std_empirical ** 2 / std_theoretical ** 2
                        chi2_cdf = chi2.cdf(chi2_value, runs - 1)
                        self.assertTrue(np.all(alpha <= chi2_cdf) and np.all(chi2_cdf <= 1.0 - alpha))

                        # Conduct chi-square test for incorrect expected standard
                        # deviations: expect failure.
                        chi2_value = (runs - 1) * std_empirical ** 2 / (1.25 * std_theoretical) ** 2
                        chi2_cdf = chi2.cdf(chi2_value, runs - 1)
                        self.assertFalse(np.all(alpha <= chi2_cdf) and np.all(chi2_cdf <= 1.0 - alpha))

                        chi2_value = (runs - 1) * std_empirical ** 2 / (0.75 * std_theoretical) ** 2
                        chi2_cdf = chi2.cdf(chi2_value, runs - 1)
                        self.assertFalse(np.all(alpha <= chi2_cdf) and np.all(chi2_cdf <= 1.0 - alpha))


class TestPrivateGradValuesAxis(unittest.TestCase):
    """Test if PrivateGradValues group data into microbatches correctly."""

    def __init__(self, methodname):
        """Initialize the test class."""
        super().__init__(methodname)

        data = np.array([[2], [3], [5], [-14], [8], [-4]], dtype=np.float32)
        w = objax.TrainVar(jn.zeros(1))

        # Squared loss: average of 0.5*(w - x)**2. Gradient is -x at w=0.
        def loss(x):
            return 0.5 * ((x - w.value) ** 2).mean()

        # Procedure to get private gradient given microbatch and l2_norm_clip.
        noise_multiplier = 0.

        def get_g(microbatch, l2_norm_clip, use_norm_accumulation, batch_axis=(0,)):
            gv_priv = objax.privacy.dpsgd.PrivateGradValues(loss, objax.VarCollection({'w': w}),
                                                            noise_multiplier, l2_norm_clip, microbatch,
                                                            batch_axis=batch_axis,
                                                            use_norm_accumulation=use_norm_accumulation)
            g_priv, v_priv = gv_priv(data)
            return g_priv

        self.get_g = get_g

    def test_private_gradvalues_axis0(self):
        """Test for batch_axis = (0,) with different microbatch."""
        for use_norm_accumulation in [True, False]:
            # microbatch = 1
            g_priv = self.get_g(microbatch=1, l2_norm_clip=1.0, use_norm_accumulation=use_norm_accumulation)
            grad_expected = -np.array([1, 1, 1, -1, 1, -1], dtype=np.float32).mean()
            np.testing.assert_allclose(g_priv[0], grad_expected, atol=1e-5, err_msg='microbatch=1')

            # microbatch = 6
            g_priv = self.get_g(microbatch=6, l2_norm_clip=1.0, use_norm_accumulation=use_norm_accumulation)
            grad_expected = 0.
            np.testing.assert_allclose(g_priv[0], grad_expected, atol=1e-5, err_msg='microbatch=6')

            # microbatch = 2
            g_priv = self.get_g(microbatch=2, l2_norm_clip=3.0, use_norm_accumulation=use_norm_accumulation)
            grad_expected = -np.array([2.5, -3, 2], dtype=np.float32).mean()  # unclipped gradient is [2.5, -4.5, 2]
            np.testing.assert_allclose(g_priv[0], grad_expected, atol=1e-5, err_msg='microbatch=2')

    def test_private_gradvalues_axis1(self):
        """Test if batch_axis = (1,) raises ValueError."""
        microbatch = 1
        l2_norm_clip = 1.0
        for use_norm_accumulation in [True, False]:
            self.assertRaises(ValueError, self.get_g, microbatch, l2_norm_clip, use_norm_accumulation, (1,))


if __name__ == '__main__':
    unittest.main()
