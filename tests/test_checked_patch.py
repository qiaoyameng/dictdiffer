# SPDX-License-Identifier: MIT

"""Tests for :func:`dictdiffer.checked_patch`."""

import copy
import unittest

import pytest

from dictdiffer import (
    PatchApplyError,
    PatchConflict,
    checked_patch,
    diff,
    patch,
)


class CheckedPatchSuccessTests(unittest.TestCase):
    """Patches that apply without any conflict."""

    def test_round_trip_matches_plain_patch(self):
        first = {
            'scalar': 1,
            'nested': {'a': [1, 2, 3], 'b': {'c': 'old'}},
            'set': {1, 2},
        }
        second = {
            'scalar': 2,
            'nested': {'a': [1, 9, 3, 4], 'b': {'c': 'new', 'd': 0}},
            'set': {2, 3},
        }
        changes = list(diff(first, second))
        result, conflicts = checked_patch(changes, first)
        assert conflicts == []
        assert result == second
        # A conflict free checked patch produces exactly the same result
        # as the legacy patcher and never touches the original.
        assert result == patch(changes, first)
        assert first != result

    def test_empty_patch_returns_copy(self):
        destination = {'a': [1, 2]}
        result, conflicts = checked_patch([], destination)
        assert conflicts == []
        assert result == destination
        assert result is not destination

    def test_in_place_returns_same_object(self):
        destination = {'a': 1}
        result, conflicts = checked_patch(
            [('change', 'a', (1, 2))], destination, in_place=True)
        assert conflicts == []
        assert result is destination
        assert destination == {'a': 2}

    def test_list_indices_are_interpreted_per_operation_stage(self):
        # Removing from the end first keeps every recorded index valid.
        changes = [
            ('remove', 'lst', [(2, 3)]),
            ('remove', 'lst', [(1, 2)]),
        ]
        result, conflicts = checked_patch(changes, {'lst': [1, 2, 3]})
        assert conflicts == []
        assert result == {'lst': [1]}

        # An earlier insertion is immediately visible to the next operation.
        changes = [
            ('add', 'lst', [(1, 9)]),
            ('change', 'lst.1', (9, 8)),
        ]
        result, conflicts = checked_patch(changes, {'lst': [1, 2, 3]})
        assert conflicts == []
        assert result == {'lst': [1, 8, 2, 3]}

    def test_child_change_then_parent_remove(self):
        changes = [
            ('change', 'a.b', (1, 2)),
            ('remove', '', [('a', {'b': 2})]),
        ]
        result, conflicts = checked_patch(changes, {'a': {'b': 1}})
        assert conflicts == []
        assert result == {}

    def test_dotted_key_is_not_split(self):
        destination = {'a.b': 1}
        result, conflicts = checked_patch(
            [('change', ['a.b'], (1, 2))], destination)
        assert conflicts == []
        assert result == {'a.b': 2}

    def test_none_is_a_real_value(self):
        result, conflicts = checked_patch(
            [('change', 'a', (None, 5))], {'a': None})
        assert conflicts == []
        assert result == {'a': 5}


class CheckedPatchConflictTests(unittest.TestCase):
    """Patches whose operations no longer match the destination."""

    def test_stale_second_operation_and_final_type_change(self):
        # Operation 0 still applies, operation 1 carries a stale old value
        # and the final operation targets a value whose type changed.
        destination = {
            'a': 1,
            'b': {'c': 2},
            'd': 'already-a-string',
        }
        changes = [
            ('change', 'a', (1, 100)),
            ('change', 'b.c', (99, 200)),
            ('change', 'd', ([10, 20], 'now-a-string')),
        ]
        original = copy.deepcopy(destination)
        result, conflicts = checked_patch(changes, destination)

        assert result is None
        assert [conflict.operation for conflict in conflicts] == [1, 2]
        assert conflicts[0].path == ('b', 'c')
        assert 'expected old value 99' in conflicts[0].reason
        assert conflicts[1].path == ('d',)
        assert 'type changed' in conflicts[1].reason
        # All locatable conflicts are reported together and the destination
        # stays untouched, even though the first operation was valid.
        assert destination == original

    def test_failure_is_atomic_in_place_too(self):
        destination = {'a': 1, 'b': 2}
        changes = [
            ('change', 'a', (1, 100)),
            ('change', 'b', (99, 200)),
        ]
        original = copy.deepcopy(destination)
        result, conflicts = checked_patch(
            changes, destination, in_place=True)
        assert result is None
        assert len(conflicts) == 1
        assert conflicts[0].operation == 1
        assert destination == original

    def test_missing_value_is_not_reported_as_none(self):
        result, conflicts = checked_patch(
            [('remove', '', [('gone', 9)])], {'a': 1})
        assert result is None
        assert len(conflicts) == 1
        conflict = conflicts[0]
        assert conflict.operation == 0
        assert conflict.path == ('gone',)
        assert 'None' not in conflict.reason
        assert 'missing' in conflict.reason

    def test_missing_change_path(self):
        result, conflicts = checked_patch(
            [('change', 'a.b', (1, 2))], {'a': 1})
        assert result is None
        assert conflicts[0].path == ('a', 'b')
        assert 'None' not in conflicts[0].reason

    def test_duplicate_change_is_detected(self):
        changes = [
            ('change', 'a', (1, 2)),
            ('change', 'a', (1, 3)),
        ]
        result, conflicts = checked_patch(changes, {'a': 1})
        assert result is None
        assert [conflict.operation for conflict in conflicts] == [1]
        assert conflicts[0].path == ('a',)

    def test_duplicate_remove_is_detected(self):
        changes = [
            ('remove', 'lst', [(1, 2)]),
            ('remove', 'lst', [(1, 2)]),
        ]
        result, conflicts = checked_patch(changes, {'lst': [1, 2, 3]})
        assert result is None
        assert len(conflicts) == 1
        assert conflicts[0].operation == 1
        assert conflicts[0].path == ('lst', 1)

    def test_duplicate_add_is_detected(self):
        changes = [
            ('add', '', [('k', 1)]),
            ('add', '', [('k', 1)]),
        ]
        result, conflicts = checked_patch(changes, {})
        assert result is None
        assert len(conflicts) == 1
        assert conflicts[0].operation == 1
        assert conflicts[0].path == ('k',)

    def test_parent_child_path_conflict(self):
        # Removing the parent makes the later child change unlocatable.
        changes = [
            ('remove', '', [('a', {'b': 1})]),
            ('change', 'a.b', (1, 2)),
        ]
        result, conflicts = checked_patch(changes, {'a': {'b': 1}})
        assert result is None
        assert len(conflicts) == 1
        assert conflicts[0].operation == 1
        assert conflicts[0].path == ('a', 'b')

    def test_container_type_change_under_node(self):
        # The dictionary at ``a`` became a list, so the string key ``b``
        # cannot be resolved anymore.
        result, conflicts = checked_patch(
            [('change', ['a', 'b'], (1, 2))], {'a': [1]})
        assert result is None
        assert conflicts[0].path == ('a', 'b')
        assert 'type changed' in conflicts[0].reason

    def test_stale_dotted_key_uses_literal_path(self):
        result, conflicts = checked_patch(
            [('change', ['a.b'], (9, 2))], {'a.b': 1})
        assert result is None
        assert conflicts[0].path == ('a.b',)

    def test_add_onto_existing_key(self):
        result, conflicts = checked_patch(
            [('add', '', [('a', 2)])], {'a': 1})
        assert result is None
        assert conflicts[0].path == ('a',)
        assert 'already exists' in conflicts[0].reason

    def test_add_list_index_out_of_range(self):
        result, conflicts = checked_patch(
            [('add', 'lst', [(5, 2)])], {'lst': [1]})
        assert result is None
        assert conflicts[0].path == ('lst', 5)
        assert 'out of range' in conflicts[0].reason

    def test_add_existing_set_member(self):
        result, conflicts = checked_patch(
            [('add', 's', [(0, {1})])], {'s': {1}})
        assert result is None
        assert conflicts[0].path == ('s', 0)

    def test_add_under_missing_or_scalar_node(self):
        result, conflicts = checked_patch(
            [('add', 'nope', [('k', 1)])], {})
        assert result is None
        assert len(conflicts) == 1

        result, conflicts = checked_patch(
            [('add', 'a', [('k', 1)])], {'a': 1})
        assert result is None
        assert 'not a container' in conflicts[0].reason

    def test_remove_with_wrong_old_value(self):
        result, conflicts = checked_patch(
            [('remove', '', [('a', 99)])], {'a': 1})
        assert result is None
        assert conflicts[0].path == ('a',)
        assert 'expected old value' in conflicts[0].reason

    def test_remove_missing_set_member(self):
        result, conflicts = checked_patch(
            [('remove', 's', [(0, {7})])], {'s': {1}})
        assert result is None
        assert 'missing' in conflicts[0].reason

    def test_change_list_index_out_of_range(self):
        result, conflicts = checked_patch(
            [('change', 'lst.5', (1, 2))], {'lst': [1]})
        assert result is None
        assert conflicts[0].path == ('lst', '5')

    def test_one_bad_entry_in_multi_entry_operation(self):
        changes = [('add', '', [('a', 1), ('b', 2)])]
        result, conflicts = checked_patch(changes, {'b': 0})
        assert result is None
        assert len(conflicts) == 1
        assert conflicts[0].operation == 0
        assert conflicts[0].path == ('b',)

    def test_unknown_action_raises_value_error(self):
        with pytest.raises(ValueError):
            checked_patch([('frobnicate', 'a', (1, 2))], {'a': 1})


class PatchConflictTests(unittest.TestCase):
    """Behaviour of the conflict descriptor."""

    def test_representation_and_equality(self):
        conflict = PatchConflict(2, ('a', 'b'), 'reason text')
        assert conflict.operation == 2
        assert conflict.path == ('a', 'b')
        assert conflict.reason == 'reason text'
        assert conflict == PatchConflict(2, ['a', 'b'], 'reason text')
        assert conflict != PatchConflict(1, ('a', 'b'), 'reason text')
        assert repr(conflict) == (
            "PatchConflict(operation=2, path=('a', 'b'), "
            "reason='reason text')")
        assert eval(repr(conflict)) == conflict

    def test_path_is_copied_into_a_tuple(self):
        path = ['a', 'b']
        conflict = PatchConflict(0, path, 'reason')
        path.append('c')
        assert conflict.path == ('a', 'b')


class CustomMappingBoundaryTests(unittest.TestCase):
    """Capability boundary with user defined containers."""

    def test_rejected_assignment_is_reported_as_conflict(self):
        class StrictMapping(dict):
            def __setitem__(self, key, value):
                raise RuntimeError('external side effect rejected')

        result, conflicts = checked_patch(
            [('add', '', [('k', 1)])], StrictMapping())
        assert result is None
        assert len(conflicts) == 1
        assert 'cannot add value' in conflicts[0].reason

    def test_in_place_apply_failure_raises_patch_apply_error(self):
        class GuardedMapping(dict):
            """Deep copyable mapping that blocks the real in place write."""

            def __init__(self, *args, reconstructing=False, **kwargs):
                super().__init__()
                self._reconstructing = reconstructing
                if args or kwargs:
                    self.update(dict(*args, **kwargs))

            def __setitem__(self, key, value):
                if self._reconstructing:
                    super().__setitem__(key, value)
                else:
                    raise RuntimeError('boom at apply')

            def __deepcopy__(self, memo):
                clone = GuardedMapping(reconstructing=True)
                memo[id(self)] = clone
                for key, value in self.items():
                    clone[copy.deepcopy(key, memo)] = copy.deepcopy(
                        value, memo)
                return clone

        destination = GuardedMapping({'a': 1})
        # Copy mode still works, because only the private working copy is
        # mutated during the check.
        result, conflicts = checked_patch(
            [('change', 'a', (1, 2))], destination)
        assert conflicts == []
        assert result == {'a': 2}
        assert destination == {'a': 1}

        # In place mode validates successfully first, then the destination's
        # own mutator rejects the write.
        with pytest.raises(PatchApplyError):
            checked_patch(
                [('change', 'a', (1, 2))], destination, in_place=True)
