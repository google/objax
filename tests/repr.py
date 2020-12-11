import functools
import unittest

import jax.numpy as jn

import objax


class TestRepr(unittest.TestCase):
    def test_seq(self):
        r = '\n'.join(['objax.nn.Sequential(',
                       '  [0] objax.nn.Linear(nin=2, nout=3, use_bias=True, w_init=xavier_normal(*, gain=1))',
                       '  [1] leaky_relu(*, negative_slope=0.01)',
                       '  [2] objax.nn.Sequential(',
                       '    [0] leaky_relu(*, negative_slope=0.5)',
                       '    [1] objax.nn.Linear(nin=4, nout=5, use_bias=True, w_init=xavier_normal(*, gain=1))',
                       '  )',
                       '  [3] tanh',
                       '  [4] objax.nn.MovingAverage(shape=(2, 3), buffer_size=4, init_value=0.2)',
                       '  [5] objax.nn.ExponentialMovingAverage(shape=(5, 6), momentum=0.999, init_value=0)',
                       '  [6] objax.nn.BatchNorm(dims=(3, 4, 5), redux=(1, 7), momentum=0.999, eps=1e-06)',
                       '  [7] objax.nn.BatchNorm0D(nin=8, momentum=0.999, eps=1e-06)',
                       '  [8] objax.nn.BatchNorm1D(nin=9, momentum=0.999, eps=1e-06)',
                       '  [9] objax.nn.BatchNorm2D(nin=19, momentum=0.999, eps=1e-06)',
                       "  [10] objax.nn.Conv2D(nin=4, nout=5, k=(3, 3), strides=(1, 1), dilations=(1, 1), groups=1,"
                       " padding='SAME', use_bias=True, w_init=kaiming_normal(*, gain=1))",
                       "  [11] objax.nn.ConvTranspose2D(nin=5, nout=4, k=(3, 3), strides=(1, 1), dilations=(1, 1),"
                       " padding='SAME', use_bias=True, w_init=kaiming_normal(*, gain=1))",
                       '  [12] objax.nn.Dropout(keep=0.7)',
                       '  [13] objax.nn.SyncedBatchNorm(dims=(3, 4, 5), redux=(1, 7), momentum=0.999, eps=1e-06)',
                       '  [14] objax.nn.SyncedBatchNorm0D(nin=8, momentum=0.999, eps=1e-06)',
                       '  [15] objax.nn.SyncedBatchNorm1D(nin=9, momentum=0.999, eps=1e-06)',
                       '  [16] objax.nn.SyncedBatchNorm2D(nin=19, momentum=0.999, eps=1e-06)',
                       ')'])
        m = objax.nn.Sequential([
            objax.nn.Linear(2, 3),
            objax.functional.leaky_relu,
            objax.nn.Sequential([
                functools.partial(objax.functional.leaky_relu, negative_slope=0.5),
                objax.nn.Linear(4, 5)]),
            objax.functional.tanh,
            objax.nn.MovingAverage((2, 3), 4, 0.2),
            objax.nn.ExponentialMovingAverage((5, 6)),
            objax.nn.BatchNorm((3, 4, 5), (1, 7)),
            objax.nn.BatchNorm0D(8),
            objax.nn.BatchNorm1D(9),
            objax.nn.BatchNorm2D(19),
            objax.nn.Conv2D(4, 5, k=3),
            objax.nn.ConvTranspose2D(4, 5, k=3),
            objax.nn.Dropout(0.7),
            objax.nn.SyncedBatchNorm((3, 4, 5), (1, 7)),
            objax.nn.SyncedBatchNorm0D(8),
            objax.nn.SyncedBatchNorm1D(9),
            objax.nn.SyncedBatchNorm2D(19),
        ])
        self.assertEqual(repr(m), r)

    def test_vars(self):
        t = objax.TrainVar(jn.zeros([1, 2, 3, 2, 1]))
        tv = '\n'.join(['objax.TrainVar(DeviceArray([[[[[0.],',
                        '                [0.]],',
                        '               [[0.],',
                        '                [0.]],',
                        '               [[0.],',
                        '                [0.]]],',
                        '              [[[0.],',
                        '                [0.]],',
                        '               [[0.],',
                        '                [0.]],',
                        '               [[0.],',
                        '                [0.]]]]], dtype=float32), reduce=reduce_mean)'])
        self.assertEqual(repr(t), tv)
        r = objax.TrainRef(t)
        rv = '\n'.join(['objax.TrainRef(ref=objax.TrainVar(DeviceArray([[[[[0.],',
                        '                [0.]],',
                        '               [[0.],',
                        '                [0.]],',
                        '               [[0.],',
                        '                [0.]]],',
                        '              [[[0.],',
                        '                [0.]],',
                        '               [[0.],',
                        '                [0.]],',
                        '               [[0.],',
                        '                [0.]]]]], dtype=float32), reduce=reduce_mean))'])
        self.assertEqual(repr(r), rv)
        t = objax.StateVar(jn.zeros([1, 2, 3, 2, 1]))
        tv = '\n'.join(['objax.StateVar(DeviceArray([[[[[0.],',
                        '                [0.]],',
                        '               [[0.],',
                        '                [0.]],',
                        '               [[0.],',
                        '                [0.]]],',
                        '              [[[0.],',
                        '                [0.]],',
                        '               [[0.],',
                        '                [0.]],',
                        '               [[0.],',
                        '                [0.]]]]], dtype=float32), reduce=reduce_mean)'])
        self.assertEqual(repr(t), tv)
        self.assertEqual(repr(objax.random.Generator().key), 'objax.RandomState(DeviceArray([0, 0], dtype=uint32))')

    def test_random(self):
        self.assertEqual(repr(objax.random.DEFAULT_GENERATOR), 'objax.random.Generator(seed=0)')

    def test_opt(self):
        self.assertEqual(repr(objax.optimizer.Adam(objax.VarCollection())),
                         'objax.optimizer.Adam(beta1=0.9, beta2=0.999, eps=1e-08)')
        self.assertEqual(repr(objax.optimizer.LARS(objax.VarCollection())),
                         'objax.optimizer.LARS(momentum=0.9, weight_decay=0.0001, tc=0.001, eps=1e-05)')
        self.assertEqual(repr(objax.optimizer.Momentum(objax.VarCollection())),
                         'objax.optimizer.Momentum(momentum=0.9, nesterov=False)')
        self.assertEqual(repr(objax.optimizer.SGD(objax.VarCollection())),
                         'objax.optimizer.SGD()')
        self.assertEqual(repr(objax.optimizer.ExponentialMovingAverage(objax.VarCollection())),
                         'objax.optimizer.ExponentialMovingAverage(momentum=0.999, debias=False, eps=1e-06)')

    def test_transform(self):
        def myloss(x):
            return (x ** 2).mean()

        g = objax.Grad(myloss, variables=objax.VarCollection(), input_argnums=(0,))
        gv = objax.GradValues(myloss, variables=objax.VarCollection(), input_argnums=(0,))
        gvp = objax.privacy.dpsgd.PrivateGradValues(myloss, objax.VarCollection(), noise_multiplier=1.,
                                                    l2_norm_clip=0.5, microbatch=1)
        self.assertEqual(repr(g), 'objax.Grad(f=myloss, input_argnums=(0,))')
        self.assertEqual(repr(gv), 'objax.GradValues(f=myloss, input_argnums=(0,))')
        self.assertEqual(repr(gvp), 'objax.privacy.dpsgd.gradient.PrivateGradValues(f=myloss, noise_multiplier=1.0,'
                                    ' l2_norm_clip=0.5, microbatch=1, batch_axis=(0,))')
        self.assertEqual(repr(objax.Jit(gv)),
                         'objax.Jit(f=objax.GradValues(f=myloss, input_argnums=(0,)), static_argnums=None)')
        self.assertEqual(repr(objax.Jit(myloss, vc=objax.VarCollection())),
                         'objax.Jit(f=objax.Function(f=myloss), static_argnums=None)')
        self.assertEqual(repr(objax.Parallel(gv)),
                         "objax.Parallel(f=objax.GradValues(f=myloss, input_argnums=(0,)),"
                         " reduce=concatenate(*, axis=0), axis_name='device', static_argnums=None)")
        self.assertEqual(repr(objax.Vectorize(myloss, vc=objax.VarCollection())),
                         'objax.Vectorize(f=objax.Function(f=myloss), batch_axis=(0,))')
        self.assertEqual(repr(objax.ForceArgs(gv, training=True, word='hello')),
                         "objax.ForceArgs(module=GradValues, training=True, word='hello')")
