# SPDX-FileCopyrightText: 2026
# SPDX-License-Identifier: MIT

"""Tests for dictdiffer.compose."""

from dictdiffer import diff, patch, revert
from dictdiffer.compose import (
    compose, ComposeConflict, UncomposablePatchError, InvalidPatchError)


def _net_is_sound(first, second, third, **kwargs):
    """compose(d1, d2) takes first to third and is revertible."""
    d1 = list(diff(first, second, **kwargs))
    d2 = list(diff(second, third, **kwargs))
    net = compose(d1, d2)
    assert patch(net, first) == third
    assert revert(net, third) == first
    assert patch(d1 + d2, first) == third
    return net


def test_identity_changes():
    assert compose([('change', 'a', (1, 2))],
                   [('change', 'a', (2, 3))]) == [('change', 'a', (1, 3))]


def test_change_reverted_cancels():
    net = compose([('change', 'a', (1, 2))], [('change', 'a', (2, 1))])
    assert net == []
    assert patch(net, {'a': 1}) == {'a': 1}


def test_add_then_delete_cancels():
    first, second, third = {'a': 1}, {'a': 1, 'b': 2}, {'a': 1}
    assert compose(diff(first, second), diff(second, third)) == []


def test_delete_then_readd_is_change_with_original_old_value():
    first, second, third = {'a': 1}, {}, {'a': 2}
    net = compose(diff(first, second), diff(second, third))
    assert net == [('change', 'a', (1, 2))]
    assert revert(net, third) == first


def test_change_then_remove_keeps_first_old_value():
    first, second, third = {'a': 1}, {'a': 2}, {}
    net = compose(diff(first, second), diff(second, third))
    assert net == [('remove', '', [('a', 1)])]
    assert revert(net, third) == first


def test_added_then_changed_value_is_final():
    first, second, third = {}, {'a': 1}, {'a': 2}
    assert compose(diff(first, second), diff(second, third)) == \
        [('add', '', [('a', 2)])]


def test_nested_dict_subpaths():
    first = {'db': {'users': {'alice': {'age': 30, 'role': 'dev'}}}}
    second = {'db': {'users': {'alice': {'age': 31, 'role': 'dev'}}}}
    third = {'db': {'users': {'alice': {'age': 31, 'role': 'admin'},
                             'bob': {'age': 40}}}}
    net = _net_is_sound(first, second, third)
    # composed change uses the original age (30), enabling revert
    assert ('change', 'db.users.alice.role', ('dev', 'admin')) in net
    assert ('change', 'db.users.alice.age', (30, 31)) in net


def test_parent_remove_after_child_edit_folds_old_subtree():
    first = {'a': {'k': 1, 'm': 2}}
    second = {'a': {'k': 1}}
    third = {}
    net = compose(diff(first, second), diff(second, third))
    assert net == [('remove', '', [('a', {'k': 1, 'm': 2})])]
    assert revert(net, third) == first


def test_list_appends_compose_and_drift():
    first = {'l': [1, 2]}
    second = {'l': [1, 2, 3]}
    third = {'l': [1, 2, 3, 4, 5]}
    net = _net_is_sound(first, second, third)
    assert net == [('add', 'l', [(2, 3), (3, 4), (4, 5)])]


def test_list_delete_then_append_becomes_change():
    first = {'l': [1, 2, 3]}
    second = {'l': [1, 2]}
    third = {'l': [1, 2, 9]}
    net = _net_is_sound(first, second, third)
    assert net == [('change', ['l', 2], (3, 9))]


def test_list_append_then_delete_cancels():
    first = {'l': [1, 2]}
    second = {'l': [1, 2, 9]}
    third = {'l': [1, 2]}
    assert compose(diff(first, second), diff(second, third)) == []


def test_list_delete_index_drift_two_phases():
    first = {'l': [1, 2, 3, 4]}
    second = {'l': [1, 2]}
    third = {'l': [1]}
    d1, d2 = list(diff(first, second)), list(diff(second, third))
    net = compose(d1, d2)
    assert net == [('remove', 'l', [(3, 4), (2, 3), (1, 2)])]
    assert patch(net, first) == third


def test_list_single_tail_delete_is_revertible():
    first = {'l': [1, 2, 3]}
    second = {'l': [1, 2]}
    third = {'l': [1, 2, 9]}
    _net_is_sound(first, second, third)


def test_list_slot_change_compose():
    first = {'l': [1, 2, 3]}
    second = {'l': [9, 2, 3, 4]}
    third = {'l': [9, 2, 3, 4, 5]}
    net = _net_is_sound(first, second, third)
    assert ('change', ['l', 0], (1, 9)) in net
    assert ('add', 'l', [(3, 4), (4, 5)]) in net


def test_list_inserted_element_edited_then_removed_cancels():
    first = {'l': [1]}
    second = {'l': [1, {'k': 1}]}
    third = {'l': [1]}
    assert compose(diff(first, second), diff(second, third)) == []


def test_list_inserted_element_mutated_is_kept_with_final_value():
    first = {'l': [1]}
    second = {'l': [1, {'k': 1}]}
    third = {'l': [1, {'k': 2}]}
    net = _net_is_sound(first, second, third)
    assert net == [('add', 'l', [(1, {'k': 2})])]


def test_set_adds_compose():
    first = {'s': {1, 2}}
    second = {'s': {1, 2, 3}}
    third = {'s': {1, 2, 3, 4}}
    net = _net_is_sound(first, second, third)
    assert net == [('add', 's', [(0, {3, 4})])]


def test_set_add_then_remove_cancels_element():
    first = {'s': {1}}
    second = {'s': {1, 2}}
    third = {'s': {1}}
    assert compose(diff(first, second), diff(second, third)) == []


def test_set_remove_then_readd_cancels():
    first = {'s': {1, 2}}
    second = {'s': {1}}
    third = {'s': {1, 2}}
    assert compose(diff(first, second), diff(second, third)) == []


def test_set_remove_then_remove_conflicts():
    d1 = [('remove', 's', [(0, {5})])]
    d2 = [('remove', 's', [(0, {5})])]
    try:
        compose(d1, d2)
    except ComposeConflict:
        pass
    else:
        raise AssertionError('expected ComposeConflict')


def test_expanded_form_supported():
    first = {'l': [1], 'a': 1}
    second = {'l': [1, 2, 3], 'a': 1, 'b': 9}
    third = {'l': [1, 2, 3, 4], 'b': 9}
    _net_is_sound(first, second, third, expand=True)


def test_list_path_notation_preserved_for_string_paths():
    first = {'a': {'x': 1}}
    second = {'a': {'x': 2}}
    third = {'a': {'x': 3}}
    net = compose(diff(first, second), diff(second, third))
    assert net == [('change', 'a.x', (1, 3))]


def test_conflict_change_removed_key():
    d1 = [('remove', '', [('x', 1)])]
    d2 = [('change', 'x', (1, 9))]
    try:
        compose(d1, d2)
    except ComposeConflict:
        pass
    else:
        raise AssertionError('expected ComposeConflict')


def test_conflict_stale_old_value():
    d1 = [('change', 'x', (1, 2))]
    d2 = [('change', 'x', (7, 9))]
    try:
        compose(d1, d2)
    except ComposeConflict:
        pass
    else:
        raise AssertionError('expected ComposeConflict')


def test_conflict_edit_below_scalar_replacement():
    d1 = [('change', 'a', ({'k': 1}, 5))]
    d2 = [('change', 'a.k', (1, 2))]
    try:
        compose(d1, d2)
    except ComposeConflict:
        pass
    else:
        raise AssertionError('expected ComposeConflict')


def test_conflict_list_delete_absent_index():
    d1 = [('remove', 'l', [(2, 3)])]
    d2 = [('remove', 'l', [(2, 3)])]
    try:
        compose(d1, d2)
    except ComposeConflict:
        pass
    else:
        raise AssertionError('expected ComposeConflict')


def test_invalid_patch_raises():
    try:
        compose([('bogus', 'a', None)], [])
    except InvalidPatchError:
        pass
    else:
        raise AssertionError('expected InvalidPatchError')


def test_integer_keyed_mapping_treated_as_list_path():
    # diff emits list paths for integer dict keys; compose cannot tell the
    # two apart and therefore processes them as list indices.
    first = {1: 'a'}
    second = {1: 'b'}
    net = compose(diff(first, second), diff(second, {1: 'c'}))
    assert net == [('change', [1], ('a', 'c'))]


def test_empty_patches():
    assert compose([], []) == []
    assert compose([], list(diff({}, {'a': 1}))) == [('add', '', [('a', 1)])]
    assert compose(list(diff({}, {'a': 1})), []) == [('add', '', [('a', 1)])]


def test_root_scalar_change_chains_symbolically():
    # compose can combine root scalar changes; applying such a tuple is a
    # separate dictdiffer limitation unrelated to composition.
    assert compose([('change', '', (1, 2))],
                   [('change', '', (2, 3))]) == [('change', '', (1, 3))]
    assert compose([('change', '', (1, 2))],
                   [('change', '', (2, 1))]) == []
