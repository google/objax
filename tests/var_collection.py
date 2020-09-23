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

"""Unittests for VarCollection."""

import re
import unittest

import jax.numpy as jn

import objax
from objax.zoo import wide_resnet


class TestVarCollection(unittest.TestCase):
    def test_init_list(self):
        """Initialize a VarCollection with a list."""
        vc = objax.VarCollection([('a', objax.TrainVar(jn.zeros(1))),
                                  ('b', objax.TrainVar(jn.ones(1)))])
        self.assertEqual(len(vc), 2)
        self.assertEqual(vc['a'].value.sum(), 0)
        self.assertEqual(vc['b'].value.sum(), 1)

    def test_init_dict(self):
        """Initialize a VarCollection with a dict."""
        vc = objax.VarCollection({'a': objax.TrainVar(jn.zeros(1)),
                                  'b': objax.TrainVar(jn.ones(1))})
        self.assertEqual(len(vc), 2)
        self.assertEqual(vc['a'].value.sum(), 0)
        self.assertEqual(vc['b'].value.sum(), 1)

    def test_weight_sharing(self):
        """Check weight sharing."""
        m = objax.ModuleList([objax.TrainVar(jn.ones(3))])
        all_vars = m.vars('m1') + m.vars('m2')
        self.assertEqual(len(all_vars), 2)
        self.assertEqual(len(list(all_vars)), 1)
        self.assertEqual(len(all_vars.tensors()), 1)
        all_vars.assign([x + 1 for x in all_vars.tensors()])
        self.assertEqual(m[0].value.sum(), 6)

    def test_name_conflict(self):
        """Check name conflict raises a ValueError."""
        vc1 = objax.VarCollection([('a', objax.TrainVar(jn.zeros(1)))])
        vc2 = objax.VarCollection([('a', objax.TrainVar(jn.ones(1)))])
        with self.assertRaises(ValueError):
            vc1 + vc2

        with self.assertRaises(ValueError):
            vc1.update(vc2)

        with self.assertRaises(ValueError):
            vc1['a'] = objax.TrainVar(jn.ones(1))

    def test_assign(self):
        vc = objax.VarCollection({'a': objax.TrainVar(jn.zeros(1))})
        vc['b'] = objax.TrainVar(jn.ones(1))
        self.assertEqual(len(vc), 2)
        self.assertEqual(vc['a'].value.sum(), 0)
        self.assertEqual(vc['b'].value.sum(), 1)

    def test_len_iter(self):
        """Verify length and iterator."""
        v1 = objax.TrainVar(jn.zeros(1))
        vshared = objax.TrainVar(jn.ones(1))
        vc1 = objax.VarCollection([('a', v1), ('b', vshared)])
        vc2 = objax.VarCollection([('c', vshared)])
        vc = vc1 + vc2
        self.assertEqual(len(vc), 3)
        self.assertEqual(len(vc.keys()), 3)
        self.assertEqual(len(vc.items()), 3)
        self.assertEqual(len(vc.values()), 3)
        self.assertEqual(len(list(vc)), 2)  # Self iterator is unique.

    def test_tensors(self):
        vshared = objax.TrainVar(jn.ones(1))
        vc = objax.VarCollection([('a', objax.TrainVar(jn.zeros(1))), ('b', vshared)])
        vc += objax.VarCollection([('c', vshared)])
        self.assertEqual(len(vc.tensors()), 2)
        self.assertEqual([x.sum() for x in vc.tensors()], [0, 1])

    def test_weight_copy(self):
        class MyModel(objax.Module):
            def __init__(self):
                self.model = wide_resnet.WideResNet(3, 10, depth=28, width=2)
                self.opt = objax.optimizer.Momentum(self.model.vars())
                self.ema = objax.optimizer.ExponentialMovingAverage(self.model.vars(), momentum=0.999)

        m1 = MyModel()
        m2 = MyModel()
        m2.vars().assign(m1.vars().tensors())
        v2 = m2.vars()
        for k, v in m1.vars().items():
            self.assertEqual(v.value.tolist(), v2[k].value.tolist(), msg=f'Variable {k} value is differing.')

    def test_rename(self):
        vc = objax.VarCollection({
            'baab': objax.TrainVar(jn.zeros(()) + 1),
            'baaab': objax.TrainVar(jn.zeros(()) + 2),
            'baaaab': objax.TrainVar(jn.zeros(()) + 3),
            'abba': objax.TrainVar(jn.zeros(()) + 4),
            'acca': objax.TrainVar(jn.zeros(()) + 5)})
        vcr = vc.rename(objax.util.Renamer({'aa': 'x', 'bb': 'y'}))
        self.assertEqual(vc['baab'], vcr['bxb'])
        self.assertEqual(vc['baaab'], vcr['bxab'])
        self.assertEqual(vc['baaaab'], vcr['bxxb'])
        self.assertEqual(vc['abba'], vcr['aya'])
        self.assertEqual(vc['acca'], vcr['acca'])

        def my_rename(x):
            return x.replace('aa', 'x').replace('bb', 'y')

        vcr = vc.rename(objax.util.Renamer(my_rename))
        self.assertEqual(vc['baab'], vcr['bxb'])
        self.assertEqual(vc['baaab'], vcr['bxab'])
        self.assertEqual(vc['baaaab'], vcr['bxxb'])
        self.assertEqual(vc['abba'], vcr['aya'])
        self.assertEqual(vc['acca'], vcr['acca'])

        vcr = vc.rename(objax.util.Renamer([(re.compile('a{2}'), 'x'), (re.compile('bb'), 'y')]))
        self.assertEqual(vc['baab'], vcr['bxb'])
        self.assertEqual(vc['baaab'], vcr['bxab'])
        self.assertEqual(vc['baaaab'], vcr['bxxb'])
        self.assertEqual(vc['abba'], vcr['aya'])
        self.assertEqual(vc['acca'], vcr['acca'])

        vcr = vc.rename(objax.util.Renamer([(re.compile('a{2}'), 'x'), (re.compile('xa'), 'y')]))
        self.assertEqual(vc['baab'], vcr['bxb'])
        self.assertEqual(vc['baaab'], vcr['byb'])
        self.assertEqual(vc['baaaab'], vcr['bxxb'])
        self.assertEqual(vc['abba'], vcr['abba'])
        self.assertEqual(vc['acca'], vcr['acca'])


if __name__ == '__main__':
    unittest.main()
