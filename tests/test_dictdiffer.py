# SPDX-FileCopyrightText: 2013 Fatih Erikli.
# SPDX-FileCopyrightText: 2013, 2014, 2015, 2016 CERN.
# SPDX-FileCopyrightText: 2017-2019 ETH Zurich, Swiss Data Science Center, Jiri Kuncar.
# SPDX-License-Identifier: MIT

import math
import unittest
from collections import OrderedDict
from collections.abc import MutableMapping, MutableSequence
from decimal import Decimal, localcontext

import pytest

from dictdiffer import (HAS_NUMPY, NumericTolerance, diff, dot_lookup,
                        patch, revert, swap)
from dictdiffer.utils import PathLimit, ToleranceMap


class DictDifferTests(unittest.TestCase):
    def test_without_dot_notation(self):
        (change1,) = diff({'a': {'x': 1}},
                          {'a': {'x': 2}},
                          dot_notation=False)

        assert change1 == ('change', ['a', 'x'], (1, 2))

    def test_with_dot_notation(self):
        (change1,) = diff({'a': {'x': 1}},
                          {'a': {'x': 2}})

        assert change1 == ('change', 'a.x', (1, 2))

    def test_addition(self):
        first = {}
        second = {'a': 'b'}
        diffed = next(diff(first, second))
        assert ('add', '', [('a', 'b')]) == diffed

    def test_deletion(self):
        first = {'a': 'b'}
        second = {}
        diffed = next(diff(first, second))
        assert ('remove', '', [('a', 'b')]) == diffed

    def test_change(self):
        first = {'a': 'b'}
        second = {'a': 'c'}
        diffed = next(diff(first, second))
        assert ('change', 'a', ('b', 'c')) == diffed

        first = {'a': None}
        second = {'a': 'c'}
        diffed = next(diff(first, second))
        assert ('change', 'a', (None, 'c')) == diffed

        first = {'a': 'c'}
        second = {'a': None}
        diffed = next(diff(first, second))
        assert ('change', 'a', ('c', None)) == diffed

        first = {'a': 'c'}
        second = {'a': u'c'}
        diffed = list(diff(first, second))
        assert [] == diffed

        first = {'a': 'b'}
        second = {'a': None}
        diffed = next(diff(first, second))
        assert ('change', 'a', ('b', None)) == diffed

        first = {'a': 10.0}
        second = {'a': 10.5}
        diffed = next(diff(first, second))
        assert ('change', 'a', (10.0, 10.5)) == diffed

    def test_immutable_diffs(self):
        first = {'a': 'a'}
        second = {'a': {'b': 'b'}}
        result = list(diff(first, second))
        assert result[0][2][1]['b'] == 'b'
        second['a']['b'] = 'c'  # result MUST stay unchanged
        assert result[0][2][1]['b'] == 'b'

    def test_tolerance(self):
        first = {'a': 'b'}
        second = {'a': 'c'}
        diffed = next(diff(first, second, tolerance=0.1))
        assert ('change', 'a', ('b', 'c')) == diffed

        first = {'a': None}
        second = {'a': 'c'}
        diffed = next(diff(first, second, tolerance=0.1))
        assert ('change', 'a', (None, 'c')) == diffed

        first = {'a': 10.0}
        second = {'a': 10.5}
        diffed = list(diff(first, second, tolerance=0.1))
        assert [] == diffed

        diffed = next(diff(first, second, tolerance=0.01))
        assert ('change', 'a', (10.0, 10.5)) == diffed

        first = {'a': 10.0, 'b': 1.0e-15}
        second = {'a': 10.5, 'b': 2.5e-15}
        diffed = sorted(diff(
            first, second, tolerance=0.01
        ))
        assert [
            ('change', 'a', (10.0, 10.5)),
            ('change', 'b', (1.0e-15, 2.5e-15)),
        ] == diffed

        diffed = sorted(diff(
            first, second, tolerance=0.01, absolute_tolerance=1e-12
        ))
        assert [('change', 'a', (10.0, 10.5))] == diffed

        diffed = sorted(diff(
            first, second, tolerance=0.01, absolute_tolerance=1e-18
        ))
        assert [
            ('change', 'a', (10.0, 10.5)),
            ('change', 'b', (1.0e-15, 2.5e-15)),
        ] == diffed

        diffed = sorted(diff(
            first, second, tolerance=0.1, absolute_tolerance=1e-18
        ))
        assert [('change', 'b', (1.0e-15, 2.5e-15))] == diffed

        diffed = sorted(diff(
            first, second, tolerance=0.1, absolute_tolerance=1e-12
        ))
        assert [] == diffed

        diffed = sorted(diff(
            first, second, tolerance=None, absolute_tolerance=None
        ))
        assert [
            ('change', 'a', (10.0, 10.5)),
            ('change', 'b', (1.0e-15, 2.5e-15)),
        ] == diffed

        first = {'a': 10.0, 'b': 1.0e-15}
        second = {'a': 10.0, 'b': 1.0e-15}
        diffed = sorted(diff(
            first, second, tolerance=None, absolute_tolerance=None
        ))
        assert [] == diffed

    def test_path_limit_as_list(self):
        first = {}
        second = {'author': {'last_name': 'Doe', 'first_name': 'John'}}
        diffed = list(diff(first, second, path_limit=[('author',)]))

        res = [('add', '', [('author',
                             {'first_name': 'John', 'last_name': 'Doe'})])]

        assert res == diffed

    def test_path_limit_addition(self):
        first = {}
        second = {'author': {'last_name': 'Doe', 'first_name': 'John'}}
        p = PathLimit([('author',)])
        diffed = list(diff(first, second, path_limit=p))

        res = [('add', '', [('author',
                             {'first_name': 'John', 'last_name': 'Doe'})])]

        assert res == diffed

        first = {}
        second = {'author': {'last_name': 'Doe', 'first_name': 'John'}}
        p = PathLimit([('author',)])
        diffed = list(diff(first, second, path_limit=p, expand=True))

        res = [('add', '', [('author',
                             {'first_name': 'John', 'last_name': 'Doe'})])]

        assert res == diffed

        first = {}
        second = {'author': {'last_name': 'Doe', 'first_name': 'John'}}
        p = PathLimit()
        diffed = list(diff(first, second, path_limit=p, expand=True))
        res = [('add', '', [('author', {})]),
               ('add', 'author', [('first_name', 'John')]),
               ('add', 'author', [('last_name', 'Doe')])]

        assert len(diffed) == 3
        for patch in res:
            assert patch in diffed

    def test_path_limit_deletion(self):
        first = {'author': {'last_name': 'Doe', 'first_name': 'John'}}
        second = {}
        p = PathLimit([('author',)])
        diffed = list(diff(first, second, path_limit=p, expand=True))

        res = [('remove', '', [('author',
                                {'first_name': 'John', 'last_name': 'Doe'})])]

        assert res == diffed

    def test_path_limit_change(self):
        first = {'author': {'last_name': 'Do', 'first_name': 'John'}}
        second = {'author': {'last_name': 'Do', 'first_name': 'John'}}
        p = PathLimit([('author',)])
        diffed = list(diff(first, second, path_limit=p, expand=True))
        assert [] == diffed

        first = {'author': {'last_name': 'Do', 'first_name': 'John'}}
        second = {'author': {'last_name': 'Doe', 'first_name': 'John'}}
        p = PathLimit([('author',)])
        diffed = list(diff(first, second, path_limit=p, expand=True))

        res = [('change',
                ['author'],
                ({'first_name': 'John', 'last_name': 'Do'},
                 {'first_name': 'John', 'last_name': 'Doe'}))]

        assert res == diffed

        first = {'author': {'last_name': 'Do', 'first_name': 'John'}}
        second = {'author': {'last_name': 'Doe', 'first_name': 'John'}}
        p = PathLimit()
        diffed = list(diff(first, second, path_limit=p, expand=True))

        res = [('change', 'author.last_name', ('Do', 'Doe'))]

        assert res == diffed

    def test_expand_addition(self):
        first = {}
        second = {'foo': 'bar', 'apple': 'banana'}
        diffed = list(diff(first, second, expand=True))
        res = [('add', '', [('foo', 'bar')]),
               ('add', '', [('apple', 'banana')])]

        assert len(diffed) == 2
        for patch in res:
            assert patch in diffed

    def test_expand_deletion(self):
        first = {'foo': 'bar', 'apple': 'banana'}
        second = {}
        diffed = list(diff(first, second, expand=True))
        res = [('remove', '', [('foo', 'bar')]),
               ('remove', '', [('apple', 'banana')])]

        assert len(diffed) == 2
        for patch in res:
            assert patch in diffed

    def test_nodes(self):
        first = {'a': {'b': {'c': 'd'}}}
        second = {'a': {'b': {'c': 'd', 'e': 'f'}}}
        diffed = next(diff(first, second))
        assert ('add', 'a.b', [('e', 'f')]) == diffed

    def test_add_list(self):
        first = {'a': []}
        second = {'a': ['b']}
        diffed = next(diff(first, second))
        assert ('add', 'a', [(0, 'b')]) == diffed

    def test_remove_list(self):
        first = {'a': ['b', 'c']}
        second = {'a': []}
        diffed = next(diff(first, second))
        assert ('remove', 'a', [(1, 'c'), (0, 'b'), ]) == diffed

    def test_add_set(self):
        first = {'a': {1, 2, 3}}
        second = {'a': {0, 1, 2, 3}}
        diffed = next(diff(first, second))
        assert ('add', 'a', [(0, set([0]))]) == diffed

    def test_remove_set(self):
        first = {'a': set([0, 1, 2, 3])}
        second = {'a': set([1, 2, 3])}
        diffed = next(diff(first, second))
        assert ('remove', 'a', [(0, set([0]))]) == diffed

    def test_change_set(self):
        first = {'a': set([0, 1, 2, 3])}
        second = {'a': set([1, 2, 3, 4])}
        diffed = list(diff(first, second))
        assert ('add', 'a', [(0, set([4]))]) in diffed
        assert ('remove', 'a', [(0, set([0]))]) in diffed

    def test_add_set_shift_order(self):
        first = set(["changeA", "changeB"])
        second = set(["changeA", "changeC", "changeB"])
        diffed = list(diff(first, second))
        # There should only be 1 change reported
        assert len(diffed) == 1
        assert ('add', '', [(0, {'changeC'})]) in diffed

    def test_change_set_order(self):
        first = set(["changeA", "changeC", "changeB"])
        second = set(["changeB", "changeC", "changeA"])
        diffed = list(diff(first, second))
        # There should be zero reported diffs
        assert len(diffed) == 0

    def test_types(self):
        first = {'a': ['a']}
        second = {'a': 'a'}
        diffed = next(diff(first, second))
        assert ('change', 'a', (['a'], 'a')) == diffed

    def test_nan(self):
        value = float('nan')
        diffed = list(diff([value], [value]))
        assert [] == diffed

        diffed = list(diff([value], [3.5]))
        assert [('change', [0], (value, 3.5))] == diffed

    @unittest.skipIf(not HAS_NUMPY, 'NumPy is not installed')
    def test_numpy_nan(self):
        """Compare NumPy NaNs (#114)."""
        import numpy as np
        first = {'a': np.float32('nan')}
        second = {'a': float('nan')}
        result = list(diff(first, second))
        assert result == []

    def test_unicode_keys(self):
        first = {u'привет': 1}
        second = {'hello': 1}
        diffed = list(diff(first, second))
        assert ('add', '', [('hello', 1)]) in diffed
        assert ('remove', '', [(u'привет', 1)]) in diffed

        diffed = list(diff(first, second, ignore=['hello']))
        assert ('remove', '', [(u'привет', 1)]) == diffed[0]

        diffed = list(diff(first, second, ignore=[u'привет']))
        assert ('add', '', [('hello', 1)]) == diffed[0]

    def test_dotted_key(self):
        first = {'a.b': {'c.d': 1}}
        second = {'a.b': {'c.d': 2}}
        diffed = list(diff(first, second))
        assert [('change', ['a.b', 'c.d'], (1, 2))] == diffed

    def test_ignore_key(self):
        first = {'a': 'a', 'b': 'b', 'c': 'c'}
        second = {'a': 'a', 'b': 2, 'c': 3}
        diffed = next(diff(first, second, ignore=['b']))
        assert ('change', 'c', ('c', 3)) == diffed

    def test_ignore_dotted_key(self):
        first = {'a': {'aa': 'A', 'ab': 'B', 'ac': 'C'}}
        second = {'a': {'aa': 1, 'ab': 'B', 'ac': 3}}
        diffed = next(diff(first, second, ignore=['a.aa']))
        assert ('change', 'a.ac', ('C', 3)) == diffed

    def test_ignore_with_unicode_sub_keys(self):
        first = {u'a': {u'aא': {u'aa': 'A'}}}
        second = {u'a': {u'aא': {u'aa': 'B'}}}

        assert len(list(diff(first, second))) == 1
        assert len(list(diff(first, second, ignore=[u'a.aא.aa']))) == 0
        assert len(
            list(diff(first, second, ignore=[[u'a', u'aא', u'aa']
                                             ]))) == 0

    def test_ignore_complex_key(self):
        first = {'a': {1: {'a': 'a', 'b': 'b'}}}
        second = {'a': {1: {'a': 1, 'b': 2}}}
        diffed = next(diff(first, second, ignore=[['a', 1, 'a']]))
        assert ('change', ['a', 1, 'b'], ('b', 2)) == diffed

    def test_ignore_missing_keys(self):
        first = {'a': 'a'}
        second = {'a': 'a', 'b': 'b'}
        assert len(list(diff(first, second, ignore=['b']))) == 0
        assert len(list(diff(second, first, ignore=['b']))) == 0

    def test_ignore_missing_complex_keys(self):
        first = {'a': {1: {'a': 'a', 'b': 'b'}}}
        second = {'a': {1: {'a': 1}}}
        diffed = next(diff(first, second, ignore=[['a', 1, 'b']]))
        assert ('change', ['a', 1, 'a'], ('a', 1)) == diffed
        diffed = next(diff(second, first, ignore=[['a', 1, 'b']]))
        assert ('change', ['a', 1, 'a'], (1, 'a')) == diffed

    def test_ignore_stringofintegers_keys(self):
        a = {'1': '1', '2': '2', '3': '3'}
        b = {'1': '1', '2': '2', '3': '99', '4': '100'}

        assert list(diff(a, b, ignore={'3', '4'})) == []

    def test_ignore_integers_keys(self):
        a = {1: 1, 2: 2, 3: 3}
        b = {1: 1, 2: 2, 3: 99, 4: 100}

        assert len(list(diff(a, b, ignore={3, 4}))) == 0

    def test_ignore_with_ignorecase(self):
        class IgnoreCase(set):
            def __contains__(self, key):
                return set.__contains__(self, str(key).lower())

        assert list(diff({'a': 1, 'b': 2}, {'A': 3, 'b': 4},
                         ignore=IgnoreCase('a'))) == [('change', 'b', (2, 4))]

    def test_complex_diff(self):
        """Check regression on issue #4."""
        from decimal import Decimal

        d1 = {
            'id': 1,
            'code': None,
            'type': u'foo',
            'bars': [
                {'id': 6934900},
                {'id': 6934977},
                {'id': 6934992},
                {'id': 6934993},
                {'id': 6935014}],
            'n': 10,
            'date_str': u'2013-07-08 00:00:00',
            'float_here': 0.454545,
            'complex': [{
                'id': 83865,
                'goal': Decimal('2.000000'),
                'state': u'active'}],
            'profile_id': None,
            'state': u'active'
        }

        d2 = {
            'id': u'2',
            'code': None,
            'type': u'foo',
            'bars': [
                {'id': 6934900},
                {'id': 6934977},
                {'id': 6934992},
                {'id': 6934993},
                {'id': 6935014}],
            'n': 10,
            'date_str': u'2013-07-08 00:00:00',
            'float_here': 0.454545,
            'complex': [{
                'id': 83865,
                'goal': Decimal('2.000000'),
                'state': u'active'}],
            'profile_id': None,
            'state': u'active'
        }

        assert len(list(diff(d1, {}))) > 0
        assert d1['id'] == 1
        assert d2['id'] == u'2'
        assert d1 is not d2
        assert d1 != d2
        assert len(list(diff(d1, d2))) > 0

    def test_list_change(self):
        """Produced diffs should not contain empty list instructions (#30)."""
        first = {"a": {"b": [100, 101, 201]}}
        second = {"a": {"b": [100, 101, 202]}}
        result = list(diff(first, second))
        assert len(result) == 1
        assert result == [('change', ['a', 'b', 2], (201, 202))]

    def test_list_same(self):
        """Diff for the same list should be empty."""
        first = {1: [1]}
        assert len(list(diff(first, first))) == 0

    @unittest.skipIf(not HAS_NUMPY, 'NumPy is not installed')
    def test_numpy_array(self):
        """Compare NumPy arrays (#68)."""
        import numpy as np
        first = np.array([1, 2, 3])
        second = np.array([1, 2, 4])
        result = list(diff(first, second))
        assert result == [('change', [2], (3, 4))]

    def test_dict_subclasses(self):
        class Foo(dict):
            pass

        first = Foo({2014: [
            dict(month=6, category=None, sum=672.00),
            dict(month=6, category=1, sum=-8954.00),
            dict(month=7, category=None, sum=7475.17),
            dict(month=7, category=1, sum=-11745.00),
            dict(month=8, category=None, sum=-12140.00),
            dict(month=8, category=1, sum=-11812.00),
            dict(month=9, category=None, sum=-31719.41),
            dict(month=9, category=1, sum=-11663.00),
        ]})

        second = Foo({2014: [
            dict(month=6, category=None, sum=672.00),
            dict(month=6, category=1, sum=-8954.00),
            dict(month=7, category=None, sum=7475.17),
            dict(month=7, category=1, sum=-11745.00),
            dict(month=8, category=None, sum=-12141.00),
            dict(month=8, category=1, sum=-11812.00),
            dict(month=9, category=None, sum=-31719.41),
            dict(month=9, category=2, sum=-11663.00),
        ]})

        diffed = next(diff(first, second))
        assert ('change', [2014, 4, 'sum'], (-12140.0, -12141.0)) == diffed

    def test_collection_subclasses(self):
        class DictA(MutableMapping):

            def __init__(self, *args, **kwargs):
                self.__dict__.update(*args, **kwargs)

            def __setitem__(self, key, value):
                self.__dict__[key] = value

            def __getitem__(self, key):
                return self.__dict__[key]

            def __delitem__(self, key):
                del self.__dict__[key]

            def __iter__(self):
                return iter(self.__dict__)

            def __len__(self):
                return len(self.__dict__)

        class DictB(MutableMapping):

            def __init__(self, *args, **kwargs):
                self.__dict__.update(*args, **kwargs)

            def __setitem__(self, key, value):
                self.__dict__[key] = value

            def __getitem__(self, key):
                return self.__dict__[key]

            def __delitem__(self, key):
                del self.__dict__[key]

            def __iter__(self):
                return iter(self.__dict__)

            def __len__(self):
                return len(self.__dict__)

        class ListA(MutableSequence):

            def __init__(self, *args, **kwargs):
                self._list = list(*args, **kwargs)

            def __getitem__(self, index):
                return self._list[index]

            def __setitem__(self, index, value):
                self._list[index] = value

            def __delitem__(self, index):
                del self._list[index]

            def __iter__(self):
                for value in self._list:
                    yield value

            def __len__(self):
                return len(self._list)

            def insert(self, index, value):
                self._list.insert(index, value)

        daa = DictA(a=ListA(['a', 'A']))
        dba = DictB(a=ListA(['a', 'A']))
        dbb = DictB(a=ListA(['b', 'A']))
        assert list(diff(daa, dba)) == []
        assert list(diff(daa, dbb)) == [('change', ['a', 0], ('a', 'b'))]
        assert list(diff(dba, dbb)) == [('change', ['a', 0], ('a', 'b'))]


class DiffPatcherTests(unittest.TestCase):
    def test_addition(self):
        first = {}
        second = {'a': 'b'}
        assert second == patch(
            [('add', '', [('a', 'b')])], first)

        first = {'a': {'b': 'c'}}
        second = {'a': {'b': 'c', 'd': 'e'}}
        assert second == patch(
            [('add', 'a', [('d', 'e')])], first)

    def test_changes(self):
        first = {'a': 'b'}
        second = {'a': 'c'}
        assert second == patch(
            [('change', 'a', ('b', 'c'))], first)

        first = {'a': {'b': {'c': 'd'}}}
        second = {'a': {'b': {'c': 'e'}}}
        assert second == patch(
            [('change', 'a.b.c', ('d', 'e'))], first)

    def test_remove(self):
        first = {'a': {'b': 'c'}}
        second = {'a': {}}
        assert second == patch(
            [('remove', 'a', [('b', 'c')])], first)

        first = {'a': 'b'}
        second = {}
        assert second == patch(
            [('remove', '', [('a', 'b')])], first)

    def test_remove_list(self):
        first = {'a': [1, 2, 3]}
        second = {'a': [1, ]}
        assert second == patch(
            [('remove', 'a', [(2, 3), (1, 2), ]), ], first)

    def test_add_list(self):
        first = {'a': [1]}
        second = {'a': [1, 2]}
        assert second == patch(
            [('add', 'a', [(1, 2)])], first)

        first = {'a': {'b': [1]}}
        second = {'a': {'b': [1, 2]}}
        assert second == patch(
            [('add', 'a.b', [(1, 2)])], first)

    def test_change_list(self):
        first = {'a': ['b']}
        second = {'a': ['c']}
        assert second == patch(
            [('change', 'a.0', ('b', 'c'))], first)

        first = {'a': {'b': {'c': ['d']}}}
        second = {'a': {'b': {'c': ['e']}}}
        assert second == patch(
            [('change', 'a.b.c.0', ('d', 'e'))], first)

        first = {'a': {'b': {'c': [{'d': 'e'}]}}}
        second = {'a': {'b': {'c': [{'d': 'f'}]}}}
        assert second == patch(
            [('change', 'a.b.c.0.d', ('e', 'f'))], first)

    def test_remove_set(self):
        first = {'a': set([1, 2, 3])}
        second = {'a': set([1])}
        assert second == patch(
            [('remove', 'a', [(0, set([2, 3]))])], first)

    def test_add_set(self):
        first = {'a': set([1])}
        second = {'a': set([1, 2])}
        assert second == patch(
            [('add', 'a', [(0, set([2]))])], first)

    def test_dict_int_key(self):
        first = {0: 0}
        second = {0: 'a'}
        first_patch = [('change', [0], (0, 'a'))]
        assert second == patch(first_patch, first)

    def test_dict_combined_key_type(self):
        first = {0: {'1': {2: 3}}}
        second = {0: {'1': {2: '3'}}}
        first_patch = [('change', [0, '1', 2], (3, '3'))]
        assert second == patch(first_patch, first)
        assert first_patch[0] == list(diff(first, second))[0]

    def test_in_place_patch_and_revert(self):
        first = {'a': 1}
        second = {'a': 2}
        changes = list(diff(first, second))
        patched_copy = patch(changes, first)
        assert first != patched_copy
        reverted_in_place = revert(changes, patched_copy, in_place=True)
        assert first == reverted_in_place
        assert patched_copy == reverted_in_place
        patched_in_place = patch(changes, first, in_place=True)
        assert first == patched_in_place


class SwapperTests(unittest.TestCase):
    def test_addition(self):
        result = 'add', '', [('a', 'b')]
        swapped = 'remove', '', [('a', 'b')]
        assert next(swap([result])) == swapped

        result = 'remove', 'a.b', [('c', 'd')]
        swapped = 'add', 'a.b', [('c', 'd')]
        assert next(swap([result])) == swapped

    def test_changes(self):
        result = 'change', '', ('a', 'b')
        swapped = 'change', '', ('b', 'a')
        assert next(swap([result])) == swapped

    def test_revert(self):
        first = {'a': [1, 2]}
        second = {'a': []}
        diffed = diff(first, second)
        patched = patch(diffed, first)
        assert patched == second
        diffed = diff(first, second)
        reverted = revert(diffed, second)
        assert reverted == first

    def test_list_of_different_length(self):
        """Check that one can revert list with different length."""
        first = [1]
        second = [1, 2, 3]
        result = list(diff(first, second))
        assert first == revert(result, second)


class ToleranceMapTest(unittest.TestCase):
    def test_strategy_validation(self):
        for bad in (-1, '1', object()):
            with pytest.raises(ValueError if isinstance(bad, int)
                               else TypeError):
                NumericTolerance(abs_tol=bad)
        with pytest.raises(TypeError):
            NumericTolerance(abs_tol=True)
        with pytest.raises(TypeError):
            NumericTolerance(abs_tol=1, nope=True)
        with pytest.raises(ValueError):
            NumericTolerance(rel_tol=float('nan'))
        with pytest.raises(ValueError):
            NumericTolerance(abs_tol=Decimal('NaN'))
        # infinite tolerances are allowed
        NumericTolerance(abs_tol=float('inf'))

    def test_path_validation(self):
        with pytest.raises(TypeError):
            ToleranceMap({'a': NumericTolerance(abs_tol=1)})
        with pytest.raises(TypeError):
            ToleranceMap({('a',): 'not-a-strategy'})
        with pytest.raises(TypeError):
            ToleranceMap({(): NumericTolerance(abs_tol=1)})

    def test_dict_values_are_converted(self):
        tm = ToleranceMap({('a',): {'abs_tol': 0.5}})
        assert tm[('a',)].abs_tol == 0.5
        assert isinstance(tm[('a',)], NumericTolerance)

    def test_wildcard_priority(self):
        tm = ToleranceMap({
            ('x', '+'): NumericTolerance(abs_tol=1),
            ('x', '*'): NumericTolerance(abs_tol=2),
            ('x', 'a'): NumericTolerance(abs_tol=3),
            ('x', 'a', '*'): NumericTolerance(abs_tol=4),
        })
        # exact > '+'
        assert tm.query(('x', 'a')).abs_tol == 3
        # '+' beats both '*' variants for a single unknown level
        assert tm.query(('x', 'b')).abs_tol == 1
        # longest '*' prefix wins
        assert tm.query(('x', 'a', 'b')).abs_tol == 4
        assert tm.query(('x', 'b', 'c')).abs_tol == 2

    def test_validation_does_not_traverse_business_objects(self):
        class Untraversable(object):
            def __eq__(self, other):
                raise AssertionError('business object was compared')

            def __iter__(self):
                raise AssertionError('business object was traversed')

            def __getitem__(self, key):
                raise AssertionError('business object was traversed')

        # Invalid strategies are rejected eagerly, before any business
        # object is inspected.
        with pytest.raises(TypeError):
            diff(Untraversable(), Untraversable(),
                 tolerances={('payload',): 'bad-strategy'})
        with pytest.raises(TypeError):
            diff(Untraversable(), Untraversable(),
                 tolerances={'payload': NumericTolerance(abs_tol=1)})


class NumericToleranceDiffTest(unittest.TestCase):
    def test_amount_and_sensor_use_separate_strategies(self):
        first = {
            'amount': {'value': 100.00},
            'sensor': {'value': 20.0001},
        }
        second = {
            'amount': {'value': 100.05},
            'sensor': {'value': 20.0004},
        }
        result = list(diff(first, second, tolerances={
            ('amount', 'value'): NumericTolerance(abs_tol=Decimal('0.01')),
            ('sensor', 'value'): NumericTolerance(abs_tol=Decimal('0.001')),
        }))
        assert result == [('change', 'amount.value',
                           (100.00, 100.05))]

    def test_absolute_and_relative_tolerance(self):
        strategy = NumericTolerance(abs_tol=0.25, rel_tol=0.001)
        # boundary: exactly at abs_tol -> unchanged
        assert list(diff({'a': 10.0}, {'a': 10.25},
                         tolerances={('a',): strategy})) == []
        # just above abs_tol -> change
        assert list(diff({'a': 10.0}, {'a': 10.26},
                         tolerances={('a',): strategy})) == [
            ('change', 'a', (10.0, 10.26))]
        # relative tolerance dominates for large values
        assert list(diff({'a': 1000.0}, {'a': 1000.8},
                         tolerances={
                             ('a',): NumericTolerance(rel_tol=0.001),
                         })) == []
        assert list(diff({'a': 1000.0}, {'a': 1001.1},
                         tolerances={
                             ('a',): NumericTolerance(rel_tol=0.001),
                         })) == [
            ('change', 'a', (1000.0, 1001.1))]

    def test_int_support(self):
        assert list(diff({'a': 100}, {'a': 101},
                         tolerances={
                             ('a',): NumericTolerance(abs_tol=1),
                         })) == []
        assert list(diff({'a': 100}, {'a': 102},
                         tolerances={
                             ('a',): NumericTolerance(abs_tol=1),
                         })) == [('change', 'a', (100, 102))]

    def test_decimal_support(self):
        strategy = NumericTolerance(abs_tol=Decimal('0.05'),
                                    rel_tol=Decimal('0.001'))
        assert list(diff({'a': Decimal('10.00')},
                         {'a': Decimal('10.05')},
                         tolerances={('a',): strategy})) == []
        assert list(diff({'a': Decimal('10.00')},
                         {'a': Decimal('10.06')},
                         tolerances={('a',): strategy})) == [
            ('change', 'a', (Decimal('10.00'), Decimal('10.06')))]
        # mixed float tolerance values are coerced for Decimal operands
        assert list(diff({'a': Decimal('10.00')},
                         {'a': Decimal('10.04')},
                         tolerances={
                             ('a',): NumericTolerance(abs_tol=0.05),
                         })) == []

    def test_bool_is_not_a_number(self):
        assert list(diff({'a': True}, {'a': 1},
                         tolerances={
                             ('a',): NumericTolerance(abs_tol=100),
                         })) == [('change', 'a', (True, 1))]
        assert list(diff({'a': True}, {'a': 1.0},
                         tolerance=10, absolute_tolerance=10)) == [
            ('change', 'a', (True, 1.0))]
        assert list(diff({'a': False}, {'a': False},
                         tolerances={
                             ('a',): NumericTolerance(abs_tol=100),
                         })) == []

    def test_special_values(self):
        # signed zeros are equal
        assert list(diff({'a': 0.0}, {'a': -0.0})) == []
        assert list(diff({'a': 0.0}, {'a': -0.0},
                         tolerances={
                             ('a',): NumericTolerance(abs_tol=0),
                         })) == []
        # NaN versus NaN follows nan_equal (default: equal)
        assert list(diff({'a': float('nan')}, {'a': float('nan')})) == []
        assert list(diff({'a': float('nan')}, {'a': float('nan')},
                         tolerances={
                             ('a',): NumericTolerance(nan_equal=True),
                         })) == []
        result = list(diff({'a': float('nan')}, {'a': float('nan')},
                           tolerances={
                               ('a',): NumericTolerance(nan_equal=False),
                           }))
        assert len(result) == 1
        action, node, values = result[0]
        assert (action, node) == ('change', 'a')
        assert math.isnan(values[0]) and math.isnan(values[1])
        # NaN versus a number is always a change, tolerance cannot hide it
        result = list(diff({'a': float('nan')}, {'a': 1.0},
                           tolerances={
                               ('a',): NumericTolerance(abs_tol=1e9),
                           }))
        assert len(result) == 1
        action, node, values = result[0]
        assert (action, node) == ('change', 'a')
        assert math.isnan(values[0]) and values[1] == 1.0
        # same-sign infinities equal, opposite signs differ
        assert list(diff({'a': float('inf')}, {'a': float('inf')})) == []
        assert list(diff({'a': float('-inf')},
                         {'a': float('-inf')})) == []
        assert list(diff({'a': float('inf')},
                         {'a': float('-inf')})) == [
            ('change', 'a', (float('inf'), float('-inf')))]
        # infinity never within tolerance of a finite value
        assert list(diff({'a': float('inf')}, {'a': 1e308},
                         tolerances={
                             ('a',): NumericTolerance(rel_tol=1.0),
                         })) == [('change', 'a', (float('inf'), 1e308))]

    def test_decimal_special_values(self):
        nan = Decimal('NaN')
        inf = Decimal('Infinity')
        assert list(diff({'a': nan}, {'a': nan})) == []
        assert list(diff({'a': nan}, {'a': nan},
                         tolerances={
                             ('a',): NumericTolerance(nan_equal=False),
                         })) == [('change', 'a', (nan, nan))]
        assert list(diff({'a': inf}, {'a': inf})) == []
        assert list(diff({'a': inf}, {'a': -inf})) == [
            ('change', 'a', (inf, -inf))]
        assert list(diff({'a': inf}, {'a': Decimal('1e300')},
                         tolerances={
                             ('a',): NumericTolerance(rel_tol=1.0),
                         })) == [('change', 'a', (inf, Decimal('1e300')))]
        # signed zeros, also for Decimal
        assert list(diff({'a': Decimal('0')}, {'a': Decimal('-0')})) == []
        # int/float and int/Decimal pairs compare numerically
        assert list(diff({'a': 10}, {'a': 10.05},
                         tolerances={
                             ('a',): NumericTolerance(abs_tol=0.1),
                         })) == []
        assert list(diff({'a': 10}, {'a': Decimal('10.05')},
                         tolerances={
                             ('a',): NumericTolerance(abs_tol=Decimal('0.1')),
                         })) == []
        # missing fields stay add/remove, never tolerance-suppressed
        assert list(diff({}, {'a': Decimal('1.0')},
                         tolerances={
                             ('a',): NumericTolerance(abs_tol=10),
                         })) == [('add', '', [('a', Decimal('1.0'))])]
        assert list(diff({'a': Decimal('1.0')}, {},
                         tolerances={
                             ('a',): NumericTolerance(abs_tol=10),
                         })) == [
            ('remove', '', [('a', Decimal('1.0'))])]

    def test_missing_field_is_not_suppressed(self):
        tolerances = {('a',): NumericTolerance(abs_tol=100)}
        assert list(diff({'b': 1}, {'b': 2, 'a': 5}, tolerances=tolerances)) \
            == [('change', 'b', (1, 2)), ('add', '', [('a', 5)])]
        assert list(diff({'a': 5, 'b': 1}, {'b': 2},
                         tolerances=tolerances)) == [
            ('change', 'b', (1, 2)), ('remove', '', [('a', 5)])]

    def test_non_numeric_values_ignored_strategy(self):
        tolerances = {('*',): NumericTolerance(abs_tol=100)}
        assert list(diff({'a': 'x'}, {'a': 'y'}, tolerances=tolerances)) \
            == [('change', 'a', ('x', 'y'))]
        assert list(diff({'a': None}, {'a': 0}, tolerances=tolerances)) \
            == [('change', 'a', (None, 0))]

    def test_strategy_wins_over_global_tolerance(self):
        # The global tolerance would suppress this change; the path
        # strategy is stricter and keeps it.
        assert list(diff({'a': 10.0}, {'a': 10.5},
                         tolerance=0.1,
                         tolerances={
                             ('a',): NumericTolerance(abs_tol=0.01),
                         })) == [('change', 'a', (10.0, 10.5))]

    def test_unmatched_path_uses_global_tolerance(self):
        assert list(diff({'a': 10.0}, {'a': 10.5},
                         tolerance=0.1,
                         tolerances={
                             ('b',): NumericTolerance(abs_tol=0.01),
                         })) == []

    def test_ignored_report(self):
        report = []
        result = list(diff({'a': 10.0, 'b': 1.0},
                          {'a': 10.5, 'b': 2.0},
                          tolerances={
                              ('a',): NumericTolerance(abs_tol=1),
                              ('b',): NumericTolerance(abs_tol=0),
                          },
                          ignored_report=report))
        assert result == [('change', 'b', (1.0, 2.0))]
        assert len(report) == 1
        node, values, strategy = report[0]
        assert node == 'a'
        assert values == (10.0, 10.5)
        assert strategy.abs_tol == 1
        # the report holds copies, not references into the inputs
        first = {'a': 10.0}
        second = {'a': 10.4}
        report = []
        list(diff(first, second,
                  tolerances={('a',): NumericTolerance(abs_tol=1)},
                  ignored_report=report))
        second['a'] = 99.0
        assert report[0][1] == (10.0, 10.4)

    def test_no_report_for_unmatched_or_non_numeric(self):
        report = []
        list(diff({'a': 10.0, 'b': 'x'}, {'a': 10.5, 'b': 'y'},
                  tolerance=0.1,
                  tolerances={('b',): NumericTolerance(abs_tol=1)},
                  ignored_report=report))
        assert report == []

    def test_no_report_for_exact_numeric_equality(self):
        report = []
        assert list(diff({'a': 1}, {'a': 1.0},
                         tolerances={('a',): NumericTolerance(abs_tol=1)},
                         ignored_report=report)) == []
        assert report == []

    def test_change_keeps_actual_target_value(self):
        first = {'a': Decimal('1.234567')}
        second = {'a': Decimal('1.234568')}
        result = list(diff(first, second,
                           tolerances={
                               ('a',): NumericTolerance(
                                   abs_tol=Decimal('0.0000001')),
                           }))
        assert result == [('change', 'a',
                           (Decimal('1.234567'), Decimal('1.234568')))]
        patched = patch(result, first)
        assert patched == second
        assert patched['a'] == Decimal('1.234568')
        assert patched['a'].as_tuple() == Decimal('1.234568').as_tuple()


class LowDecimalPrecisionToleranceTest(unittest.TestCase):
    def test_low_precision_diff_and_patch(self):
        with localcontext() as ctx:
            ctx.prec = 4
            strategy = NumericTolerance(abs_tol=Decimal('0.5'))
            first = {'value': Decimal('1.000')}
            second = {'value': Decimal('1.400')}
            assert list(diff(first, second,
                             tolerances={('value',): strategy})) == []
            # beyond the threshold the change is reported with the
            # unrounded values and patched verbatim
            second = {'value': Decimal('1.600')}
            result = list(diff(first, second,
                               tolerances={('value',): strategy}))
            assert result == [('change', 'value',
                               (Decimal('1.000'), Decimal('1.600')))]
            assert patch(result, first)['value'] == Decimal('1.600')

    def test_low_precision_threshold_boundary(self):
        with localcontext() as ctx:
            ctx.prec = 3
            first = {'v': Decimal('9.99')}
            # 9.99 + 0.01 == 10.0 at prec 3, exactly at abs_tol
            assert list(diff(first, {'v': Decimal('10.0')},
                             tolerances={
                                 ('v',): NumericTolerance(
                                     abs_tol=Decimal('0.01')),
                             })) == []
            assert list(diff(first, {'v': Decimal('10.1')},
                             tolerances={
                                 ('v',): NumericTolerance(
                                     abs_tol=Decimal('0.01')),
                             })) == [
                ('change', 'v', (Decimal('9.99'), Decimal('10.1')))]

    def test_low_precision_relative_boundary(self):
        with localcontext() as ctx:
            ctx.prec = 4
            strategy = NumericTolerance(rel_tol=Decimal('0.01'))
            # rel boundary: 1% of 100.0 == 1.0
            assert list(diff({'v': Decimal('100.0')},
                             {'v': Decimal('101.0')},
                             tolerances={('v',): strategy})) == []
            assert list(diff({'v': Decimal('100.0')},
                             {'v': Decimal('101.1')},
                             tolerances={('v',): strategy})) == [
                ('change', 'v', (Decimal('100.0'), Decimal('101.1')))]


class DotLookupTest(unittest.TestCase):
    def test_list_lookup(self):
        source = {0: '0'}
        assert dot_lookup(source, [0]) == '0'

    def test_invalit_lookup_type(self):
        self.assertRaises(TypeError, dot_lookup, {0: '0'}, 0)


@pytest.mark.parametrize(
    'ignore,dot_notation,diff_size', [
        (u'nifi.zookeeper.session.timeout', True, 1),
        (u'nifi.zookeeper.session.timeout', False, 0),
        ((u'nifi.zookeeper.session.timeout', ), True, 0),
        ((u'nifi.zookeeper.session.timeout', ), False, 0),
    ],
)
def test_ignore_dotted_ignore_key(ignore, dot_notation, diff_size):
    key_to_ignore = u'nifi.zookeeper.session.timeout'
    config_dict = OrderedDict(
        [('address', 'devops011-slv-01.gvs.ggn'),
            (key_to_ignore, '3 secs')])

    ref_dict = OrderedDict(
        [('address', 'devops011-slv-01.gvs.ggn'),
            (key_to_ignore, '4 secs')])

    assert diff_size == len(
        list(diff(config_dict, ref_dict,
                  dot_notation=dot_notation,
                  ignore=[ignore])))


if __name__ == "__main__":
    unittest.main()
