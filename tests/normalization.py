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

"""Unittests for Batch Normalization Layer."""
import os
import unittest

import jax.numpy as jn
import numpy as np

import objax


# Split CPU cores into 8 devices for tests of sync batch norm
os.environ['XLA_FLAGS'] = ' '.join(os.environ.get('XLA_FLAGS', '').split(' ')
                                   + ['--xla_force_host_platform_device_count=8'])


class TestBatchnorm(unittest.TestCase):

    def test_batchnorm_0d(self):
        x = objax.random.normal((64, 8))
        bn = objax.nn.BatchNorm0D(8)
        # run batch norm in training mode
        yt = bn(x, training=True)
        self.assertEqual(yt.shape, x.shape)
        # run batch norm in eval mode
        ye = bn(x, training=False)
        self.assertEqual(ye.shape, x.shape)

    def test_batchnorm_1d(self):
        x = objax.random.normal((64, 4, 16))
        bn = objax.nn.BatchNorm1D(4)
        # run batch norm in training mode
        yt = bn(x, training=True)
        self.assertEqual(yt.shape, x.shape)
        # run batch norm in eval mode
        ye = bn(x, training=False)
        self.assertEqual(ye.shape, x.shape)

    def test_batchnorm_2d(self):
        x = objax.random.normal((64, 3, 16, 16))
        bn = objax.nn.BatchNorm2D(3)
        # run batch norm in training mode
        yt = bn(x, training=True)
        self.assertEqual(yt.shape, x.shape)
        # run batch norm in eval mode
        ye = bn(x, training=False)
        self.assertEqual(ye.shape, x.shape)


class TestSyncBatchnorm(unittest.TestCase):

    def assertTensorsAlmostEqual(self, a, b):
        a = np.array(a)
        b = np.array(b)
        np.testing.assert_almost_equal(a, b, decimal=5)

    def assertTensorsNotEqual(self, a, b):
        a = np.array(a)
        b = np.array(b)
        self.assertGreater(((a - b) ** 2).sum(), 1e-5)

    def helper_test_syncbn(self, x, bn_fn, syncbn_fn):
        # run regular batch norm in train and eval mode
        bn = bn_fn()
        yt = bn(x, training=True)
        ye = bn(x, training=False)
        # run replicated sync batch norm in train and eval mode
        sync_bn = syncbn_fn()
        sync_bn_train = objax.Parallel(lambda x: sync_bn(x, training=True), vc=sync_bn.vars())
        sync_bn_eval = objax.Parallel(lambda x: sync_bn(x, training=False), vc=sync_bn.vars())
        with sync_bn.vars().replicate():
            yt_syncbn = sync_bn_train(x)
            ye_syncbn = sync_bn_eval(x)
        # replicated sync bn should have the same behavior as non-replicated regular bn
        self.assertTensorsAlmostEqual(yt, yt_syncbn)
        self.assertTensorsAlmostEqual(ye, ye_syncbn)
        self.assertTensorsAlmostEqual(yt, yt_syncbn)
        self.assertTensorsAlmostEqual(bn.running_mean.value, sync_bn.running_mean.value)
        self.assertTensorsAlmostEqual(bn.running_var.value, sync_bn.running_var.value)
        # run replicated non-sync batch norm - it should yield different result
        non_sync_bn = bn_fn()
        non_sync_bn_train = objax.Parallel(lambda x: non_sync_bn(x, training=True), vc=non_sync_bn.vars())
        non_sync_bn_eval = objax.Parallel(lambda x: non_sync_bn(x, training=False), vc=non_sync_bn.vars())
        with non_sync_bn.vars().replicate():
            yt_non_syncbn = non_sync_bn_train(x)
            ye_non_syncbn = non_sync_bn_eval(x)
        self.assertTensorsNotEqual(yt, yt_non_syncbn)
        self.assertTensorsNotEqual(ye, ye_non_syncbn)

    def test_syncbn_0d(self):
        x = objax.random.normal((64, 8))
        self.helper_test_syncbn(x, lambda: objax.nn.BatchNorm0D(8), lambda: objax.nn.SyncedBatchNorm0D(8))

    def test_syncbn_1d(self):
        x = objax.random.normal((64, 4, 16))
        self.helper_test_syncbn(x, lambda: objax.nn.BatchNorm1D(4), lambda: objax.nn.SyncedBatchNorm1D(4))

    def test_syncbn_2d(self):
        x = objax.random.normal((64, 3, 16, 16))
        self.helper_test_syncbn(x, lambda: objax.nn.BatchNorm2D(3), lambda: objax.nn.SyncedBatchNorm2D(3))


class TestGroupnorm(unittest.TestCase):

    def test_groupnorm_0d(self):
        x = objax.random.normal((64, 8))
        gn = objax.nn.GroupNorm0D(8, groups=2, eps=1e-8)
        y = gn(x, training=True)
        self.assertEqual(y.shape, x.shape)
        x = jn.array([[1., 1., 1., 1., -1., -2., -3., -4.], [5., 5., 5., 5., -1., -1., -2., -2.]])
        y = gn(x)
        np.testing.assert_allclose(
            np.array([[0., 0., 0., 0., 1.34164079, 0.4472136, -0.4472136, -1.34164079],
                      [0., 0., 0., 0., 1., 1., -1., -1.]]),
            y,
            atol=1e-5)

    def test_groupnorm_1d(self):
        x = objax.random.normal((64, 8, 16))
        gn = objax.nn.GroupNorm1D(8, groups=2)
        y = gn(x, training=True)
        self.assertEqual(y.shape, x.shape)

    def test_groupnorm_2d(self):
        x = objax.random.normal((64, 4, 16, 16))
        gn = objax.nn.GroupNorm2D(4, groups=2)
        y = gn(x, training=True)
        self.assertEqual(y.shape, x.shape)


if __name__ == '__main__':
    unittest.main()
