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

"""Unittests for Convolution Layer."""
import unittest

import numpy as np
import scipy.stats

import objax


class TestNNInit(unittest.TestCase):
    def test_kaiming_normal(self):
        """Kaiming He normal."""
        for s in ((4, 3, 2, 10000), (12, 2, 10000), (24, 10000), (32, 10000)):
            for gain in (1, 2):
                init = np.asarray(objax.nn.init.kaiming_normal(s, gain=gain))
                std = np.sqrt(1 / np.prod(s[:-1]))
                self.assertAlmostEqual(init.mean(), 0, delta=1e-2, msg=(s, gain))
                self.assertAlmostEqual(init.std(), std * gain, delta=1e-2, msg=(s, gain))

    def test_kaiming_truncated_normal(self):
        """Kaiming He truncated normal."""
        for s in ((4, 3, 2, 10000), (12, 2, 10000), (24, 10000), (32, 10000)):
            for gain in (1, 2):
                for a, b in ((-100, 100), (-2, 2), (-0.1, 0.1)):
                    init = np.asarray(objax.nn.init.kaiming_truncated_normal(s, lower=a, upper=b, gain=gain))
                    std = np.sqrt(1 / np.prod(s[:-1]))
                    truncated_std = scipy.stats.truncnorm.std(a=a, b=b, loc=0., scale=1)
                    self.assertGreaterEqual(init.min() + 1e-6, a * std * gain / truncated_std, msg=(s, gain, (a, b)))
                    self.assertLessEqual(init.max() - 1e-6, b * std * gain / truncated_std, msg=(s, gain, (a, b)))
                    self.assertAlmostEqual(init.mean(), 0, delta=1e-2, msg=(s, gain, (a, b)))
                    self.assertAlmostEqual(init.std(), std * gain, delta=1e-2, msg=(s, gain, (a, b)))

    def test_truncated_normal(self):
        """Truncated normal."""
        for s in ((4, 3, 2, 10000), (12, 2, 10000), (24, 10000), (32, 10000)):
            for std in (1, 2):
                for a, b in ((-100, 100), (-2, 2), (-0.1, 0.1)):
                    init = np.asarray(objax.nn.init.truncated_normal(s, lower=a, upper=b, stddev=std))
                    truncated_std = scipy.stats.truncnorm.std(a=a, b=b, loc=0., scale=1)
                    self.assertGreaterEqual(init.min() + 1e-6, a * std / truncated_std, msg=(s, std, (a, b)))
                    self.assertLessEqual(init.max() - 1e-6, b * std / truncated_std, msg=(s, std, (a, b)))
                    self.assertAlmostEqual(init.mean(), 0, delta=1e-2, msg=(s, std, (a, b)))
                    self.assertAlmostEqual(init.std(), std, delta=1e-2, msg=(s, std, (a, b)))

    def test_xavier_normal(self):
        """Xavier Glorot normal."""
        for s in ((4, 3, 2, 10000), (12, 2, 10000), (24, 10000), (32, 10000)):
            for gain in (1, 2):
                init = np.asarray(objax.nn.init.xavier_normal(s, gain=gain))
                std = np.sqrt(2 / (np.prod(s[:-1]) + s[-1]))
                self.assertAlmostEqual(init.mean(), 0, delta=1e-2, msg=(s, gain))
                self.assertAlmostEqual(init.std(), std * gain, delta=1e-2, msg=(s, gain))

    def test_xavier_truncated_normal(self):
        """Xavier Glorot truncated normal."""
        for s in ((4, 3, 2, 10000), (12, 2, 10000), (24, 10000), (32, 10000)):
            for gain in (1, 2):
                for a, b in ((-100, 100), (-2, 2), (-0.1, 0.1)):
                    init = np.asarray(objax.nn.init.xavier_truncated_normal(s, lower=a, upper=b, gain=gain))
                    std = np.sqrt(2 / (np.prod(s[:-1]) + s[-1]))
                    truncated_std = scipy.stats.truncnorm.std(a=a, b=b, loc=0., scale=1)
                    self.assertGreaterEqual(init.min() + 1e-6, a * std * gain / truncated_std, msg=(s, gain, (a, b)))
                    self.assertLessEqual(init.max() - 1e-6, b * std * gain / truncated_std, msg=(s, gain, (a, b)))
                    self.assertAlmostEqual(init.mean(), 0, delta=1e-2, msg=(s, gain, (a, b)))
                    self.assertAlmostEqual(init.std(), std * gain, delta=1e-2, msg=(s, gain, (a, b)))

    def test_identity(self):
        """Identity"""
        for s in ((16, 16), (24, 10000), (10000, 32)):
            for gain in (1, 2):
                init = np.asarray(objax.nn.init.identity(s, gain=gain))
                diff_sum = np.linalg.norm(init - gain * np.eye(s[0], s[1]))
                self.assertAlmostEqual(diff_sum, 0, delta=1e-2, msg=(s, gain))

    def test_orthogonal(self):
        """Orthogonal."""
        for s in ((10, 100), (100, 10), (10, 10)):
            for gain in (1, 2):
                init = np.asarray(objax.nn.init.orthogonal(s, gain=gain))
                I = init@init.T if s[0] < s[1] else init.T@init
                I /= gain**2
                diff = I - np.eye(min(s[0], s[1]))
                self.assertAlmostEqual(np.linalg.norm(diff), 0, delta=1e-2, msg=(s, gain))
    
    def test_kaiming_normal_gain(self):
        """Kaiming normal gain."""
        shapes = ((4, 3, 2, 10000), (12, 2, 10000), (24, 10000), (32, 10000))
        expected_gain = (0.20412415, 0.20412415, 0.20412415, 0.17677670)
        for s, egain in zip(shapes, expected_gain):
            gain = objax.nn.init.kaiming_normal_gain(s)
            self.assertAlmostEqual(gain, egain, delta=1e-6, msg=(s, gain, (gain, egain)))

    def test_xavier_normal_gain(self):
        """Xavier Glorot normal gain."""
        shapes = ((4, 3, 2, 10000), (12, 2, 10000), (24, 10000), (32, 10000))
        expected_gain = (0.01412520, 0.01412520, 0.01412520, 0.01411956)
        for s, egain in zip(shapes, expected_gain):
            gain = objax.nn.init.xavier_normal_gain(s)
            self.assertAlmostEqual(gain, egain, delta=1e-6, msg=(s, gain, (gain, egain)))

    def test_gain_leaky_relu(self):
        """Leaky ReLU gain."""
        slopes = (0.01, 0.1, 0.5, 1)
        expected_gain = (1.41414286, 1.40719509, 1.26491106, 1.00000000)
        for s, egain in zip(slopes, expected_gain):
            gain = objax.nn.init.gain_leaky_relu(s)
            self.assertAlmostEqual(gain, egain, delta=1e-6, msg=(s, gain, (gain, egain)))


if __name__ == '__main__':
    unittest.main()
