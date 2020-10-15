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

"""Unittests for Parallel Layer."""
import os
import unittest

import jax.numpy as jn
import numpy as np

import objax

# Split CPU cores into 8 devices for tests of objax.Parallel
os.environ['XLA_FLAGS'] = ' '.join(os.environ.get('XLA_FLAGS', '').split(' ')
                                   + ['--xla_force_host_platform_device_count=8'])


class TestParallel(unittest.TestCase):

    def test_parallel_concat(self):
        """Parallel inference (concat reduction)."""
        f = objax.nn.Linear(3, 4)
        x = objax.random.normal((96, 3))
        y = f(x)
        fp = objax.Parallel(f)
        with fp.vars().replicate():
            z = fp(x)
        self.assertTrue(jn.array_equal(y, z))

    def test_parallel_concat_multi_output(self):
        """Parallel inference (concat reduction) for multiple outputs."""
        f = objax.nn.Linear(3, 4)
        x = objax.random.normal((96, 3))
        y = f(x)
        fp = objax.Parallel(lambda x: [f(x), f(-x)], vc=f.vars())
        with fp.vars().replicate():
            z1, z2 = fp(x)
        self.assertTrue(jn.array_equal(z1, y))
        self.assertTrue(jn.array_equal(z2, -y))

    def test_parallel_bneval_concat(self):
        """Parallel inference (concat reduction) with batch norm in eval mode."""
        f = objax.nn.Sequential([objax.nn.Conv2D(3, 4, k=1), objax.nn.BatchNorm2D(4), objax.nn.Conv2D(4, 2, k=1)])
        x = objax.random.normal((96, 3, 16, 16))
        y = f(x, training=False)
        fp = objax.Parallel(lambda x: f(x, training=False), f.vars())
        with fp.vars().replicate():
            z = fp(x)
        self.assertTrue(jn.array_equal(z, y))
        self.assertEqual(((f[1].running_var.value - 1) ** 2).sum(), 0)
        self.assertEqual((f[1].running_mean.value ** 2).sum(), 0)

    def test_parallel_bntrain_concat(self):
        """Parallel inference (concat reduction) with batch norm in train mode."""
        f = objax.nn.Sequential([objax.nn.Conv2D(3, 4, k=1), objax.nn.BatchNorm2D(4), objax.nn.Conv2D(4, 2, k=1)])
        x = objax.random.normal((64, 3, 16, 16))
        states, values = [], []
        tensors = f.vars().tensors()
        for it in range(2):
            values.append(f(x, training=True))
            states += [f[1].running_var.value, f[1].running_mean.value]
        f.vars().assign(tensors)

        self.assertAlmostEqual(((values[0] - values[1]) ** 2).sum(), 0, places=8)
        self.assertGreater(((states[0] - states[2]) ** 2).sum(), 0)
        self.assertGreater(((states[1] - states[3]) ** 2).sum(), 0)

        fp8 = objax.Parallel(lambda x: f(x, training=True), vc=f.vars())
        x8 = jn.broadcast_to(x, (8, 64, 3, 16, 16)).reshape((-1, 3, 16, 16))
        tensors = fp8.vars().tensors()
        for it in range(2):
            with fp8.vars().replicate():
                z = fp8(x8).reshape((-1,) + values[it].shape)
            self.assertAlmostEqual(((f[1].running_var.value - states[2 * it]) ** 2).sum(), 0, delta=1e-12)
            self.assertAlmostEqual(((f[1].running_mean.value - states[2 * it + 1]) ** 2).sum(), 0, delta=1e-12)
            self.assertLess(((z - values[it][None]) ** 2).sum(), 1e-7)
        fp8.vars().assign(tensors)

    def test_parallel_syncbntrain_concat(self):
        """Parallel inference (concat reduction) with synced batch norm in train mode."""
        f = objax.nn.Sequential([objax.nn.Conv2D(3, 4, k=1), objax.nn.BatchNorm2D(4), objax.nn.Conv2D(4, 2, k=1)])
        fs = objax.nn.Sequential(f[:])
        fs[1] = objax.nn.SyncedBatchNorm2D(4)
        x = objax.random.normal((96, 3, 16, 16))
        states, values = [], []
        tensors = f.vars().tensors()
        for it in range(2):
            values.append(f(x, training=True))
            states += [f[1].running_var.value, f[1].running_mean.value]
        f.vars().assign(tensors)

        self.assertAlmostEqual(((values[0] - values[1]) ** 2).sum(), 0, places=8)
        self.assertGreater(((states[0] - states[2]) ** 2).sum(), 0)
        self.assertGreater(((states[1] - states[3]) ** 2).sum(), 0)

        fp = objax.Parallel(lambda x: fs(x, training=True), vc=fs.vars())
        for it in range(2):
            with fp.vars().replicate():
                z = fp(x)
            self.assertAlmostEqual(((fs[1].running_var.value - states[2 * it]) ** 2).sum(), 0, delta=1e-12)
            self.assertAlmostEqual(((fs[1].running_mean.value - states[2 * it + 1]) ** 2).sum(), 0, delta=1e-12)
            self.assertLess(((z - values[it]) ** 2).sum(), 1e-7)

    def test_parallel_train_op(self):
        """Parallel train op."""
        f = objax.nn.Sequential([objax.nn.Linear(3, 4), objax.functional.relu, objax.nn.Linear(4, 2)])
        centers = objax.random.normal((1, 2, 3))
        centers *= objax.functional.rsqrt((centers ** 2).sum(2, keepdims=True))
        x = (objax.random.normal((256, 1, 3), stddev=0.1) + centers).reshape((512, 3))
        y = jn.concatenate([jn.zeros((256, 1), dtype=jn.uint32), jn.ones((256, 1), dtype=jn.uint32)], axis=1)
        y = y.reshape((512,))
        opt = objax.optimizer.Momentum(f.vars())
        all_vars = f.vars('f') + opt.vars('opt')

        def loss(x, y):
            xe = objax.functional.loss.cross_entropy_logits_sparse(f(x), y)
            return xe.mean()

        gv = objax.GradValues(loss, f.vars())

        def train_op(x, y):
            g, v = gv(x, y)
            opt(0.05, g)
            return v

        tensors = all_vars.tensors()
        loss_value = np.array([train_op(x, y)[0] for _ in range(10)])
        var_values = {k: v.value for k, v in all_vars.items()}
        all_vars.assign(tensors)

        self.assertGreater(loss_value.min(), 0)

        def train_op_para(x, y):
            g, v = gv(x, y)
            opt(0.05, objax.functional.parallel.pmean(g))
            return objax.functional.parallel.pmean(v)

        fp = objax.Parallel(train_op_para, vc=all_vars, reduce=lambda x: x[0])
        with all_vars.replicate():
            loss_value_p = np.array([fp(x, y)[0] for _ in range(10)])
        var_values_p = {k: v.value for k, v in all_vars.items()}

        self.assertLess(jn.abs(loss_value_p / loss_value - 1).max(), 1e-6)
        for k, v in var_values.items():
            self.assertLess(((v - var_values_p[k]) ** 2).sum(), 1e-12, msg=k)

    def test_parallel_weight_decay(self):
        """Parallel weight decay."""
        f = objax.nn.Sequential([objax.nn.Linear(3, 4), objax.nn.Linear(4, 2)])
        fvars = f.vars()

        def loss_fn():
            return 0.5 * sum((v.value ** 2).sum() for k, v in fvars.items() if k.endswith('.w'))

        tensors = fvars.tensors()
        loss_value = loss_fn()
        fvars.assign(tensors)
        self.assertGreater(loss_value, 0)

        fp = objax.Parallel(loss_fn, vc=fvars, reduce=lambda x: x[0])
        with fvars.replicate():
            loss_value_p = fp()
        self.assertLess(abs(loss_value_p / loss_value - 1), 1e-6)

    def test_parallel_ema(self):
        """Parallel EMA."""
        f = objax.nn.Sequential([objax.nn.Linear(3, 4), objax.nn.BatchNorm0D(4),
                                 objax.nn.Linear(4, 2), objax.nn.Dropout(1)])
        ema = objax.optimizer.ExponentialMovingAverage(f.vars().subset(objax.TrainVar))
        ema_f = ema.replace_vars(f)
        ema()
        all_vars = f.vars() + ema.vars()

        fp = objax.Parallel(lambda x: f(x, training=False), f.vars())
        ema_fp = objax.Parallel(lambda x: ema_f(x, training=False), all_vars)
        x = objax.random.normal((96, 3))
        with all_vars.replicate():
            y = fp(x)
            z = ema_fp(x)
        self.assertGreater(jn.abs(y - z).mean(), 1e-3)

    def test_parallel_ema_shared(self):
        """Parallel EMA with weight sharing."""
        f = objax.nn.Sequential([objax.nn.Linear(3, 4), objax.nn.BatchNorm0D(4),
                                 objax.nn.Linear(4, 2), objax.nn.Dropout(1)])
        ema = objax.optimizer.ExponentialMovingAverage(f.vars().subset(objax.TrainVar))
        ema_f = ema.replace_vars(f)
        ema()
        all_vars = f.vars() + ema.vars()

        ema_fp = objax.Parallel(lambda x: ema_f(x, training=False), all_vars)
        ema_fps = objax.Parallel(lambda x: ema_f(x, training=False), all_vars + f.vars('shared'))
        x = objax.random.normal((96, 3))
        with all_vars.replicate():
            z = ema_fp(x)
            zs = ema_fps(x)
        self.assertLess(jn.abs(z - zs).mean(), 1e-6)

    def test_parallel_reducers(self):
        """Parallel reductions."""
        f = objax.nn.Linear(3, 4)
        x = objax.random.normal((96, 3))
        y = f(x)
        zl = []
        for reduce in (lambda x: x,
                       lambda x: x[0],
                       lambda x: x.mean(0),
                       lambda x: x.sum(0)):
            fp = objax.Parallel(f, reduce=reduce)
            with fp.vars().replicate():
                zl.append(fp(x))
        znone, zfirst, zmean, zsum = zl
        self.assertAlmostEqual(jn.square(jn.array(y.split(8)) - znone).sum(), 0, places=8)
        self.assertAlmostEqual(jn.square(y.split(8)[0] - zfirst).sum(), 0, places=8)
        self.assertAlmostEqual(jn.square(np.mean(y.split(8), 0) - zmean).sum(), 0, places=8)
        self.assertAlmostEqual(jn.square(np.sum(y.split(8), 0) - zsum).sum(), 0, places=8)

    def test_jit_parallel_bntrain_concat(self):
        """JIT parallel inference (concat reduction) with batch norm in train mode."""
        f = objax.nn.Sequential([objax.nn.Conv2D(3, 4, k=1), objax.nn.BatchNorm2D(4), objax.nn.Conv2D(4, 2, k=1)])
        x = objax.random.normal((64, 3, 16, 16))
        states, values = [], []
        tensors = f.vars().tensors()
        for it in range(2):
            values.append(f(x, training=True))
            states += [f[1].running_var.value, f[1].running_mean.value]
        f.vars().assign(tensors)

        self.assertAlmostEqual(((values[0] - values[1]) ** 2).sum(), 0, places=8)
        self.assertGreater(((states[0] - states[2]) ** 2).sum(), 0)
        self.assertGreater(((states[1] - states[3]) ** 2).sum(), 0)

        fp = objax.Jit(objax.Parallel(lambda x: f(x, training=True), vc=f.vars()))
        x8 = jn.broadcast_to(x, (8, 64, 3, 16, 16)).reshape((-1, 3, 16, 16))
        tensors = fp.vars().tensors()
        for it in range(2):
            with fp.vars().replicate():
                z = fp(x8).reshape((-1,) + values[it].shape)
            self.assertAlmostEqual(((f[1].running_var.value - states[2 * it]) ** 2).sum(), 0, delta=1e-12)
            self.assertAlmostEqual(((f[1].running_mean.value - states[2 * it + 1]) ** 2).sum(), 0, delta=1e-12)
            self.assertLess(((z - values[it][None]) ** 2).sum(), 1e-6)
        fp.vars().assign(tensors)

    def test_jit_parallel_reducers(self):
        """JIT parallel reductions."""
        f = objax.nn.Linear(3, 4)
        x = objax.random.normal((96, 3))
        y = f(x)
        zl = []
        for reduce in (lambda x: x,
                       lambda x: x[0],
                       lambda x: x.mean(0),
                       lambda x: x.sum(0)):
            fp = objax.Jit(objax.Parallel(f, reduce=reduce))
            with fp.vars().replicate():
                zl.append(fp(x))
        znone, zfirst, zmean, zsum = zl
        self.assertAlmostEqual(jn.square(jn.array(y.split(8)) - znone).sum(), 0, places=8)
        self.assertAlmostEqual(jn.square(y.split(8)[0] - zfirst).sum(), 0, places=8)
        self.assertAlmostEqual(jn.square(np.mean(y.split(8), 0) - zmean).sum(), 0, places=8)
        self.assertAlmostEqual(jn.square(np.sum(y.split(8), 0) - zsum).sum(), 0, places=8)


if __name__ == '__main__':
    unittest.main()
