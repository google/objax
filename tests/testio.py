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

"""Unittests for IO operations."""

import io
import tempfile
import unittest

import jax.numpy as jn

import objax
from objax.zoo import wide_resnet


class TestIO(unittest.TestCase):
    def test_file_load_save_var_collection(self):
        a = objax.nn.Conv2D(16, 16, 3)
        b = objax.nn.Conv2D(16, 16, 3)
        self.assertFalse(jn.array_equal(a.w.value, b.w.value))
        with io.BytesIO() as f:
            objax.io.save_var_collection(f, a.vars())
            f.seek(0)
            objax.io.load_var_collection(f, b.vars())
        self.assertEqual(a.w.value.dtype, b.w.value.dtype)
        self.assertTrue(jn.array_equal(a.w.value, a.w.value))

    def test_filename_load_save_var_collection(self):
        a = objax.nn.Conv2D(16, 16, 3)
        b = objax.nn.Conv2D(16, 16, 3)
        self.assertFalse(jn.array_equal(a.w.value, b.w.value))
        with tempfile.NamedTemporaryFile('wb') as f:
            objax.io.save_var_collection(f.name, a.vars())
            objax.io.load_var_collection(f.name, b.vars())
        self.assertEqual(a.w.value.dtype, b.w.value.dtype)
        self.assertTrue(jn.array_equal(a.w.value, a.w.value))

    def test_file_load_save_references(self):
        a = objax.nn.Conv2D(16, 16, 3)
        b = objax.nn.Conv2D(16, 16, 3)
        c = objax.nn.Conv2D(16, 16, 3)
        refs = objax.ModuleList([objax.TrainRef(a.w)])
        crefs = objax.ModuleList([objax.TrainRef(c.w)])
        self.assertFalse(jn.array_equal(a.w.value, b.w.value))
        with io.BytesIO() as f:
            objax.io.save_var_collection(f, a.vars())
            size = f.tell()
            f.seek(0)
            objax.io.save_var_collection(f, a.vars() + refs.vars())
            self.assertEqual(size, f.tell())
            f.seek(0)
            objax.io.load_var_collection(f, b.vars())
            f.seek(0)
            with self.assertRaises(ValueError):
                objax.io.load_var_collection(f, refs.vars() + c.vars())
            f.seek(0)
            objax.io.load_var_collection(f, crefs.vars() + c.vars())
        self.assertEqual(a.w.value.dtype, b.w.value.dtype)
        self.assertEqual(a.w.value.dtype, c.w.value.dtype)
        self.assertTrue(jn.array_equal(a.w.value, b.w.value))
        self.assertTrue(jn.array_equal(a.w.value, c.w.value))

    def test_real_case(self):
        class MyModel(objax.Module):
            def __init__(self):
                self.model = wide_resnet.WideResNet(3, 10, depth=28, width=2)
                self.opt = objax.optimizer.Momentum(self.model.vars())
                self.ema = objax.optimizer.ExponentialMovingAverage(self.model.vars(), momentum=0.999)

        m1 = MyModel()
        m2 = MyModel()
        with io.BytesIO() as f:
            objax.io.save_var_collection(f, m1.vars())
            f.seek(0)
            objax.io.load_var_collection(f, m2.vars())

        v2 = m2.vars()
        for k, v in m1.vars().items():
            self.assertEqual(v.value.tolist(), v2[k].value.tolist(), msg=f'Variable {k} value is differing.')

    def test_file_load_var_collection_rename(self):
        a = objax.nn.Conv2D(16, 16, 3)
        b = objax.nn.Conv2D(16, 16, 3)
        self.assertFalse(jn.array_equal(a.w.value, b.w.value))
        with io.BytesIO() as f:
            objax.io.save_var_collection(f, a.vars().rename(objax.util.Renamer({'(Conv2D)': '(MyConv2D)'})))
            f.seek(0)
            objax.io.load_var_collection(f, b.vars(), renamer=objax.util.Renamer({'(MyConv2D)': '(Conv2D)'}))
        self.assertEqual(a.w.value.dtype, b.w.value.dtype)
        self.assertTrue(jn.array_equal(a.w.value, a.w.value))


class TestCheckpoint(unittest.TestCase):
    def test_save_load_checkpoint(self):
        class MyModel(objax.Module):
            def __init__(self, depth=16, width=2):
                self.model = wide_resnet.WideResNet(3, 10, depth=depth, width=width)
                self.opt = objax.optimizer.Momentum(self.model.vars())
                self.ema = objax.optimizer.ExponentialMovingAverage(self.model.vars(), momentum=0.999)

        m1 = MyModel(depth=28, width=2)
        m2 = MyModel(depth=28, width=2)

        # make sure that variables are different right after initialization
        var_abs_diff = 0
        v2 = m2.vars()
        for k, v in m1.vars().items():
            var_abs_diff += jn.sum(jn.abs(v.value - v2[k].value))
        self.assertGreater(var_abs_diff, 1e-3)

        with tempfile.TemporaryDirectory() as logdir:
            ckpt = objax.io.checkpoint.Checkpoint(logdir, keep_ckpts=2)
            ckpt.save(m1.vars(), 1)
            ckpt.save(m1.vars(), 2)
            ckpt.save(m1.vars(), 3)
            restore_idx, _ = ckpt.restore(m2.vars())
            self.assertEqual(restore_idx, 3)

        v2 = m2.vars()
        for k, v in m1.vars().items():
            self.assertEqual(v.value.tolist(), v2[k].value.tolist(), msg=f'Variable {k} value is differing.')

        m3 = MyModel(depth=28, width=1)
            
        with tempfile.TemporaryDirectory() as logdir:
            ckpt = objax.io.checkpoint.Checkpoint(logdir, keep_ckpts=2)
            ckpt.save(m1.vars(), 1)
            with self.assertRaises(AssertionError):
                ckpt.restore(m3.vars())
            
        m3 = MyModel(depth=16, width=2)
            
        with tempfile.TemporaryDirectory() as logdir:
            ckpt = objax.io.checkpoint.Checkpoint(logdir, keep_ckpts=2)
            ckpt.save(m1.vars(), 1)
            with self.assertRaises(AssertionError):
                ckpt.restore(m3.vars())
            
                


if __name__ == '__main__':
    unittest.main()
