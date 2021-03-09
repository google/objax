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

import unittest

import numpy as np

import objax


class TestScheduler(unittest.TestCase):
    def test_linear_annealing(self):
        lin = objax.optimizer.scheduler.LinearAnnealing(max_step=10, base_lr=1, is_cycle=True, min_lr=0.1)
        lrs = []
        for i in range(10):
            lrs.append(lin())
        lrs_gt = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        np.testing.assert_array_almost_equal(lrs, lrs_gt)

    def test_step_decay(self):
        lin = objax.optimizer.scheduler.StepDecay(step_size=3, base_lr=1, gamma=0.9)
        lrs = []
        for i in range(10):
            lrs.append(lin())
        lrs_gt = [1, 1, 1, 0.9, 0.9, 0.9, 0.81, 0.81, 0.81, 0.729]
        np.testing.assert_array_almost_equal(lrs, lrs_gt)

    def test_multi_step_decay(self):
        lin = objax.optimizer.scheduler.StepDecay(step_size=[3, 5, 8], base_lr=1, gamma=0.9)
        lrs = []
        for i in range(10):
            lrs.append(lin())
        lrs_gt = [1, 1, 1, 0.9, 0.9, 0.81, 0.81, 0.81, 0.729, 0.729]
        np.testing.assert_array_almost_equal(lrs, lrs_gt)


if __name__ == '__main__':
    unittest.main()
