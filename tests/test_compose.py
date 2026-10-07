# SPDX-FileCopyrightText: 2026 TU Wien.
# SPDX-License-Identifier: MIT

import pytest

from dictdiffer import (PatchCompositionError, compose, diff, patch, revert,
                        swap)


@pytest.mark.parametrize('expand', [False, True])
def test_compose_matches_sequential_patching(expand):
    first = {
        'scalar': 1,
        'nested': {'x': 1, 'y': [0, 1, 2]},
        'members': {1, 2},
    }
    middle = {
        'scalar': 2,
        'nested': {'x': 1, 'y': [2, 1, 3], 'z': 'new'},
        'members': {2, 3},
    }
    final = {
        'scalar': 3,
        'nested': {'x': 4, 'y': [3, 1, 4]},
        'members': {3, 4},
        'later': True,
    }

    first_patch = list(diff(first, middle, expand=expand))
    second_patch = list(diff(middle, final, expand=expand))
    composed = compose(first_patch, second_patch)

    sequential = patch(second_patch, patch(first_patch, first))
    assert patch(composed, first) == final
    assert patch(composed, first) == sequential
    assert revert(composed, final) == first


def test_change_then_restore_is_cancelled():
    first_patch = [('change', 'a', (1, 2))]
    second_patch = [('change', 'a', (2, 1))]

    assert compose(first_patch, second_patch) == []


def test_add_then_remove_is_cancelled():
    first_patch = [('add', '', [('x', {'value': [1, 2]})])]
    second_patch = [
        ('change', ['x', 'value', 0], (1, 3)),
        ('remove', '', [('x', {'value': [3, 2]})]),
    ]

    composed = compose(first_patch, second_patch)
    assert composed == []


def test_list_insert_remove_index_drift_and_restore():
    first = {'items': ['a', 'b', 'c']}
    middle = {'items': ['x', 'a', 'b', 'd', 'c']}
    final = {'items': ['x', 'a', 'd', 'c']}

    composed = compose(
        list(diff(first, middle, expand=True)),
        list(diff(middle, final, expand=True)))

    assert patch(composed, first) == final
    assert revert(composed, final) == first
    assert patch(composed, first) == final


def test_set_operations_compose_and_cancel():
    first_patch = [
        ('add', 's', [(0, {3})]),
        ('remove', 's', [(0, {1})]),
    ]
    second_patch = [
        ('add', 's', [(0, {4})]),
        ('remove', 's', [(0, {2})]),
    ]

    composed = compose(first_patch, second_patch)
    assert composed == [
        ('remove', 's', [(0, {1, 2})]),
        ('add', 's', [(0, {3, 4})]),
    ]
    assert patch(composed, {'s': {1, 2}}) == {'s': {3, 4}}
    assert revert(composed, {'s': {3, 4}}) == {'s': {1, 2}}

    assert compose(
        [('add', 's', [(0, {1})])],
        [('remove', 's', [(0, {1})])]) == []


def test_dictionary_subpath_removal_and_readdition():
    first_patch = [
        ('change', 'd.x', (1, 2)),
        ('add', 'd', [('z', 9)]),
    ]
    second_patch = [
        ('remove', 'd', [('x', 2)]),
        ('add', 'd', [('y', 3)]),
        ('remove', 'd', [('z', 9)]),
    ]

    composed = compose(first_patch, second_patch)
    assert composed == [
        ('remove', 'd', [('x', 1)]),
        ('add', 'd', [('y', 3)]),
    ]


@pytest.mark.parametrize('first_patch,second_patch', [
    ([('change', 'a', (1, 2))], [('change', 'a', (9, 3))]),
    ([('add', '', [('a', 1)])], [('remove', '', [('a', 2)])]),
    ([('remove', 'items', [(0, 'a')])],
     [('change', ['items', 0], ('b', 'c'))]),
    ([('add', 's', [(0, {1})])], [('add', 's', [(0, {1})])]),
])
def test_conflicting_preconditions_are_reported(first_patch, second_patch):
    with pytest.raises(PatchCompositionError) as exc:
        compose(first_patch, second_patch)
    assert exc.value.second_patch is not None or \
        exc.value.first_patch is not None


def test_unsupported_root_replacement():
    with pytest.raises(PatchCompositionError):
        compose([('change', '', (1, 2))], [('change', '', (2, 3))])


def test_swap_restores_grouped_full_list_removal():
    patch_result = [('remove', '', [(2, 3), (1, 2), (0, 1)])]

    assert patch(swap(patch_result), []) == [1, 2, 3]


def test_invalid_patch_shape():
    with pytest.raises(PatchCompositionError):
        compose([('change', 'a')], [])
