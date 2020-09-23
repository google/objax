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

"""Unittests for objax.util."""

import re
import unittest

import objax


class TestUtil(unittest.TestCase):
    def test_easy_dict(self):
        d = objax.util.EasyDict(a=5, b=6)
        self.assertEqual(d, dict(a=5, b=6))
        self.assertEqual(d, objax.util.EasyDict({'a': 5, 'b': 6}))
        self.assertEqual(d, objax.util.EasyDict([('a', 5), ('b', 6)]))
        self.assertEqual(list(d.items()), [('a', 5), ('b', 6)])
        self.assertEqual(list(d.values()), [5, 6])
        self.assertEqual(list(d.keys()), ['a', 'b'])
        self.assertEqual([d.a, d.b], [5, 6])

    def test_renamer(self):
        r = objax.util.Renamer({'aa': 'x', 'bb': 'y'})
        self.assertEqual(r('baab'), 'bxb')
        self.assertEqual(r('baaab'), 'bxab')
        self.assertEqual(r('baaaab'), 'bxxb')
        self.assertEqual(r('abba'), 'aya')
        self.assertEqual(r('acca'), 'acca')

        def my_rename(x):
            return x.replace('aa', 'x').replace('bb', 'y')

        r = objax.util.Renamer(my_rename)
        self.assertEqual(r('baab'), 'bxb')
        self.assertEqual(r('baaab'), 'bxab')
        self.assertEqual(r('baaaab'), 'bxxb')
        self.assertEqual(r('abba'), 'aya')
        self.assertEqual(r('acca'), 'acca')

        r = objax.util.Renamer([(re.compile('a{2}'), 'x'), (re.compile('bb'), 'y')])
        self.assertEqual(r('baab'), 'bxb')
        self.assertEqual(r('baaab'), 'bxab')
        self.assertEqual(r('baaaab'), 'bxxb')
        self.assertEqual(r('abba'), 'aya')
        self.assertEqual(r('acca'), 'acca')

        r = objax.util.Renamer([(re.compile('a{2}'), 'x'), (re.compile('xa'), 'y')])
        self.assertEqual(r('baaab'), 'byb')

    def test_args_indexes(self):
        """Test args_indexes"""

        def f1(a, b, c, d=1, *, e=2, f=3, **kwargs):
            pass

        def f2(a, b, c, *args, e=2, f=3, **kwargs):
            pass

        self.assertEqual(list(objax.util.args_indexes(f1, ['a', 'c', 'd'])), [0, 2, 3])
        self.assertEqual(list(objax.util.args_indexes(f2, ['a', 'c'])), [0, 2])
        with self.assertRaises(ValueError):
            list(objax.util.args_indexes(f1, ['f']))
        with self.assertRaises(ValueError):
            list(objax.util.args_indexes(f2, ['f']))

    def test_ilog2(self):
        """Test ilog2"""
        for x in (0, 1, 2, 3, 16):
            self.assertEqual(objax.util.ilog2(1 << x), x)
            self.assertEqual(objax.util.ilog2((1 << x) + 0.01), x + 1)
            self.assertEqual(objax.util.ilog2((2 << x) - 0.01), x + 1)

    def test_positional_args_names(self):
        """Test positional_args_names"""

        def f1(a, b, c, d=1, *, e=2, f=3, **kwargs):
            pass

        def f2(a, b, c, *args, e=2, f=3, **kwargs):
            pass

        self.assertEqual(objax.util.positional_args_names(f1), ['a', 'b', 'c', 'd'])
        self.assertEqual(objax.util.positional_args_names(f2), ['a', 'b', 'c'])

    def test_to_tuple(self):
        """Test to_tuple"""
        self.assertEqual(objax.util.to_tuple((1, 2), 3), (1, 2))
        self.assertEqual(objax.util.to_tuple(1, 3), (1, 1, 1))
        self.assertEqual(objax.util.to_tuple(1.5, 2), (1.5, 1.5))
        self.assertEqual(objax.util.to_tuple(1.5, 3), (1.5, 1.5, 1.5))
        self.assertEqual(objax.util.to_tuple([1, 2], 3), (1, 2))


if __name__ == '__main__':
    unittest.main()
