# SPDX-FileCopyrightText: 2026 dictdiffer contributors.
# SPDX-License-Identifier: MIT

"""Tests for :func:`dictdiffer.compose`.

The guiding property exercised here is::

    patch(compose(p, q), A) == patch(q, patch(p, A))
    revert(compose(p, q), C) == A

while ``compose`` itself never receives ``A`` (it is only handed the two
patch iterators), so nothing can be answered by fabricating a default base.
"""

from copy import deepcopy
import random

import pytest
from dictdiffer import (
    ComposeConflict,
    ComposeUnknown,
    compose,
    diff,
    patch,
    revert,
)


def _roundtrip(a, b, c, **kwargs):
    """Compose the two real diffs and assert patch/revert equivalence."""
    p = list(diff(a, b))
    q = list(diff(b, c))
    composed = list(compose(iter(p), iter(q), **kwargs))
    assert patch(composed, a) == c
    assert revert(composed, c) == a
    # Sequential application is the reference semantics.
    assert patch(q, patch(p, a)) == c
    return composed


class TestScalarDictionaryRules:
    def test_change_chain(self):
        p = [('change', 'a', (1, 2))]
        q = [('change', 'a', (2, 3))]
        assert list(compose(p, q)) == [('change', 'a', (1, 3))]

    def test_add_then_change(self):
        p = [('add', '', [('a', 1)])]
        q = [('change', 'a', (1, 2))]
        assert list(compose(p, q)) == [('add', '', [('a', 2)])]

    def test_change_then_remove(self):
        p = [('change', 'a', (1, 2))]
        q = [('remove', '', [('a', 2)])]
        assert list(compose(p, q)) == [('remove', '', [('a', 1)])]

    def test_add_then_remove_cancels(self):
        p = [('add', '', [('a', 1)])]
        q = [('remove', '', [('a', 1)])]
        assert list(compose(p, q)) == []

    def test_remove_then_change_conflicts(self):
        p = [('remove', '', [('a', 1)])]
        q = [('change', 'a', (1, 2))]
        with pytest.raises(ComposeConflict):
            list(compose(p, q))

    def test_remove_then_remove_conflicts(self):
        p = [('remove', '', [('a', 1)])]
        q = [('remove', '', [('a', 1)])]
        with pytest.raises(ComposeConflict):
            list(compose(p, q))

    def test_add_then_add_conflicts(self):
        p = [('add', '', [('a', 1)])]
        q = [('add', '', [('a', 2)])]
        with pytest.raises(ComposeConflict):
            list(compose(p, q))

    def test_contradictory_change_is_conflict_not_guess(self):
        p = [('change', 'a', (1, 2))]
        q = [('change', 'a', (99, 3))]
        with pytest.raises(ComposeConflict):
            list(compose(p, q))

    def test_add_then_remove_with_wrong_value_conflicts(self):
        p = [('add', '', [('a', 1)])]
        q = [('remove', '', [('a', 2)])]
        with pytest.raises(ComposeConflict):
            list(compose(p, q))

    def test_unrelated_keys_pass_through(self):
        p = [('change', 'a', (1, 2))]
        q = [('change', 'b', (10, 20))]
        result = list(compose(p, q))
        assert ('change', 'a', (1, 2)) in result
        assert ('change', 'b', (10, 20)) in result


class TestNestedDictionaries:
    def test_nested_change_chain(self):
        a = {'d': {'x': 1}}
        b = {'d': {'x': 2}}
        c = {'d': {'x': 3}}
        assert _roundtrip(a, b, c) == [('change', 'd.x', (1, 3))]

    def test_add_container_then_modify_inside(self):
        a = {}
        b = {'d': {'x': 1}}
        c = {'d': {'x': 1, 'y': 2}}
        composed = _roundtrip(a, b, c)
        assert composed == [('add', '', [('d', {'x': 1, 'y': 2})])]

    def test_modify_inside_then_remove_container(self):
        a = {'d': {'x': 1}}
        b = {'d': {'x': 2}}
        c = {}
        composed = _roundtrip(a, b, c)
        assert composed == [('remove', '', [('d', {'x': 1})])]

    def test_replace_container_then_modify_inside(self):
        a = {'d': {'x': 1}}
        b = {'d': {'x': 2}}
        c = {'d': {'x': 2, 'y': 3}}
        composed = _roundtrip(a, b, c)
        assert composed == [
            ('change', 'd', ({'x': 1}, {'x': 2, 'y': 3}))]

    def test_deeply_nested_mixed_ops(self):
        a = {'d': {'sub': {'keep': 0, 'v': 1}, 'gone': 9}}
        b = {'d': {'sub': {'keep': 0, 'v': 2}, 'gone': 9, 'new': 5}}
        c = {'d': {'sub': {'keep': 7, 'v': 2}, 'new': 5}}
        _roundtrip(a, b, c)

    def test_add_then_modify_newly_added_nested_value(self):
        p = [('add', '', [('d', {'x': 1})])]
        q = [('change', 'd.x', (1, 2))]
        assert list(compose(p, q)) == [
            ('add', '', [('d', {'x': 2})])]

    def test_modify_then_delete_recovers_original(self):
        p = [('change', 'd.x', (1, 2))]
        q = [('remove', 'd', [('x', 2)])]
        composed = list(compose(p, q))
        assert composed == [('remove', 'd', [('x', 1)])]
        assert revert(composed, {}) == {'d': {'x': 1}}


class TestLists:
    def test_change_only_short_list(self):
        a = [10, 20, 30]
        b = [10, 21, 30]
        c = [10, 21, 32]
        assert _roundtrip(a, b, c) == [
            ('change', [1], (20, 21)), ('change', [2], (30, 32))]

    def test_add_then_change_inserted_element(self):
        a = {'l': [1]}
        b = {'l': [1, 2]}
        c = {'l': [1, 9]}
        _roundtrip(a, b, c)

    def test_remove_then_insert_at_same_position(self):
        a = {'l': [1, 2, 3]}
        b = {'l': [1, 3]}
        c = {'l': [1, 7, 3]}
        _roundtrip(a, b, c)

    def test_insert_head_shifts_later_change(self):
        a = {'l': [1, 2]}
        b = {'l': [0, 1, 2]}
        c = {'l': [0, 1, 8]}
        _roundtrip(a, b, c)

    def test_position_move(self):
        # Move an element from the tail to the head across the two revisions.
        a = {'l': ['a', 'b', 'c']}
        b = {'l': ['a', 'b']}
        c = {'l': ['b', 'a']}
        _roundtrip(a, b, c)

    def test_repeated_positional_operations(self):
        a = {'l': [1, 2, 3]}
        b = {'l': [1, 2]}
        c = {'l': [9, 1, 2, 4]}
        composed = _roundtrip(a, b, c)
        # Reapplying the composed patch must stay stable even though the
        # underlying sequence saw several inserts/deletes at index 0.
        assert patch(composed, a) == c

    def test_insert_removed_in_second_patch_cancels(self):
        a = {'l': [1]}
        b = {'l': [1, 2]}
        c = {'l': [1]}
        assert _roundtrip(a, b, c) == []

    def test_change_inserted_slot_then_keep(self):
        a = {'l': [1]}
        b = {'l': [1, 2]}
        c = {'l': [1, 2, 3]}
        _roundtrip(a, b, c)

    def test_nested_list_inside_dict(self):
        a = {'d': {'l': [1, 2, 3]}}
        b = {'d': {'l': [1, 3]}}
        c = {'d': {'l': [3, 1]}}
        _roundtrip(a, b, c)

    def test_root_list_with_changes_only(self):
        a = [10, 20, 30]
        b = [10, 21, 30]
        c = [10, 21, 32]
        assert _roundtrip(a, b, c) == [
            ('change', [1], (20, 21)), ('change', [2], (30, 32))]


class TestSets:
    def test_set_add_remove_composition(self):
        a = {'s': {1, 2}}
        b = {'s': {1, 2, 3}}
        c = {'s': {2, 3}}
        _roundtrip(a, b, c)

    def test_set_readd_removed_conflicts(self):
        p = [('remove', 's', [(0, {1})])]
        q = [('add', 's', [(0, {1})])]
        with pytest.raises(ComposeConflict):
            list(compose(p, q))


class TestIteratorsAndUnknown:
    def test_one_shot_generators(self):
        def gen(items):
            for item in items:
                yield item

        p = gen([('change', 'a', (1, 2))])
        q = gen([('change', 'a', (2, 3))])
        assert list(compose(p, q)) == [('change', 'a', (1, 3))]

    def test_verify_raises_unknown_for_unverifiable_precondition(self):
        # q modifies a key p never mentioned; under verification the
        # pre-condition cannot be derived from the patches.
        p = [('change', 'a', (1, 2))]
        q = [('change', 'b', (5, 6))]
        with pytest.raises(ComposeUnknown) as exc:
            list(compose(p, q, verify=True))
        assert exc.value.path == ('b',)

    def test_prior_resolves_unknown(self):
        p = [('change', 'a', (1, 2))]
        q = [('change', 'b', (5, 6))]
        result = list(compose(p, q, prior={('b',): 5}, verify=True))
        assert ('change', 'b', (5, 6)) in result

    def test_unknown_is_distinct_from_conflict(self):
        p = [('change', 'a', (1, 2))]
        q_unknown = [('change', 'b', (5, 6))]
        q_conflict = [('change', 'a', (99, 3))]
        with pytest.raises(ComposeUnknown):
            list(compose(p, q_unknown, verify=True))
        with pytest.raises(ComposeConflict):
            list(compose(p, q_conflict))

    def test_list_precondition_unknown_then_resolved(self):
        # An insertion precedes a change to a base slot the first patch never
        # established.  By default q's own old value is trusted; under
        # verification the caller must supply that intermediate (B) value,
        # keyed by its B position.
        p = [('add', 'l', [(0, 9)])]
        q = [('change', 'l.3', (30, 31))]
        base = {'l': [10, 20, 30]}
        assert patch(p, base)['l'] == [9, 10, 20, 30]

        trusted = list(compose(p, q))
        assert patch(trusted, base) == {'l': [9, 10, 20, 31]}

        with pytest.raises(ComposeUnknown):
            list(compose(p, q, verify=True))
        resolved = list(compose(
            p, q, verify=True, prior={('l', 3): 30}))
        assert patch(resolved, base) == {'l': [9, 10, 20, 31]}
        assert revert(resolved, {'l': [9, 10, 20, 31]}) == base


class TestComposeEquivalenceProperty:
    """Drive compose from real diffs on random short nested structures."""

    def _values(self, rng, depth=0):
        if depth >= 2:
            return rng.choice([0, 1, 2, 'x', 'y', None])
        kind = rng.choice(['dict', 'list', 'scalar'])
        if kind == 'scalar':
            return rng.choice([0, 1, 2, 3])
        if kind == 'list':
            return [rng.choice([0, 1, 2, 3])
                    for _ in range(rng.randint(0, 4))]
        keys = rng.sample(['a', 'b', 'c', 'd'], rng.randint(0, 3))
        return {key: self._values(rng, depth + 1) for key in keys}

    def _mutate(self, obj, rng, depth=0):
        obj = deepcopy(obj)
        if isinstance(obj, dict):
            if not obj or rng.random() < 0.4:
                obj[rng.choice(['a', 'b', 'c', 'd'])] = \
                    self._values(rng, depth + 1)
            if obj and rng.random() < 0.4:
                del obj[rng.choice(list(obj))]
            for key in list(obj):
                if rng.random() < 0.5:
                    obj[key] = self._mutate(obj[key], rng, depth + 1)
        elif isinstance(obj, list):
            if rng.random() < 0.5:
                obj.insert(rng.randint(0, len(obj)), rng.choice([0, 1, 9]))
            if obj and rng.random() < 0.5:
                del obj[rng.randrange(len(obj))]
            if obj and rng.random() < 0.5:
                obj[rng.randrange(len(obj))] = rng.choice([4, 5, 6])
        else:
            return rng.choice([0, 1, 2, 3])
        return obj

    def test_many_random_revisions(self):
        rng = random.Random(20260716)
        for _ in range(200):
            a = self._values(rng)
            b = self._mutate(a, rng)
            c = self._mutate(b, rng)
            p = list(diff(a, b))
            q = list(diff(b, c))
            composed = list(compose(iter(p), iter(q)))
            assert patch(composed, a) == c, (a, b, c, composed)
            assert revert(composed, c) == a, (a, b, c, composed)

    def test_many_random_root_list_revisions(self):
        rng = random.Random(99173)
        for _ in range(100):
            a = [rng.choice([0, 1, 2, 3]) for _ in range(rng.randint(0, 5))]
            b = self._mutate(a, rng)
            c = self._mutate(b, rng)
            p = list(diff(a, b))
            q = list(diff(b, c))
            composed = list(compose(iter(p), iter(q)))
            assert patch(composed, a) == c, (a, b, c, composed)
            assert revert(composed, c) == a, (a, b, c, composed)
