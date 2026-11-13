# SPDX-License-Identifier: MIT

"""Tests for migrating stored dictdiffer patches to renamed field paths."""

import pytest

from dictdiffer import diff, patch, revert, swap
from dictdiffer.migration import (
    MigrationError,
    migrate_document,
    migrate_patch,
    migrate_patches,
)


OLD_DOC = {
    'user': {
        'name': 'Ada',
        'email': 'ada@example.com',
        'address': {'city': 'London'},
        'tags': ['admin', 'user'],
        'labels': {'a': 1, 'b': 2},
    },
}

NEW_DOC = {
    'account': {
        'display_name': 'Ada',
        'contact_email': 'ada@example.com',
        'location': {'city': 'London'},
        'tags': ['admin', 'user'],
        'labels': {'b': 1, 'a': 2},
    },
}

MAPPING = {
    ('user',): ('account',),
    ('user', 'name'): ('account', 'display_name'),
    ('user', 'email'): ('account', 'contact_email'),
    ('user', 'address'): ('account', 'location'),
    ('user', 'labels', 'a'): ('account', 'labels', 'b'),
    ('user', 'labels', 'b'): ('account', 'labels', 'a'),
}


def test_documents_are_equivalent_under_mapping():
    assert migrate_document(OLD_DOC, MAPPING) == NEW_DOC


def test_old_patch_then_migrated_document_matches_migrated_patch_then_apply():
    edited_old = {
        'user': {
            'name': 'Ada Lovelace',
            'email': 'ada@new.example.com',
            'address': {'city': 'Manchester', 'zip': 'M1'},
            'tags': ['admin', 'user', 'math'],
            'labels': {'a': 10, 'b': 20, 'c': 30},
        },
    }
    old_patches = list(
        diff(OLD_DOC, edited_old, dot_notation=False, expand=False))
    assert old_patches

    migrated_patches = migrate_patches(old_patches, MAPPING)
    migrated_base = migrate_document(OLD_DOC, MAPPING)
    expected = migrate_document(edited_old, MAPPING)

    route_a = migrate_document(patch(old_patches, OLD_DOC), MAPPING)
    route_b = patch(migrated_patches, migrated_base)

    assert route_b == expected
    assert route_a == route_b


def test_migrated_patches_equal_native_diff_between_migrated_documents():
    edited_old = {
        'user': {
            'name': 'Grace',
            'email': 'grace@example.com',
            'address': {'city': 'New York'},
            'tags': ['admin'],
            'labels': {'a': 7},
        },
    }
    old_patches = list(diff(OLD_DOC, edited_old, dot_notation=False))
    native_new_patches = list(
        diff(migrate_document(OLD_DOC, MAPPING),
             migrate_document(edited_old, MAPPING),
             dot_notation=False))

    assert sorted(map(repr, migrate_patches(old_patches, MAPPING))) == \
        sorted(map(repr, native_new_patches))


def test_swap_and_revert_stay_equivalent_after_migration():
    edited_old = {
        'user': {
            'name': 'Katherine',
            'email': 'katherine@example.com',
            'address': {'city': 'Hampton'},
            'tags': ['admin', 'user'],
            'labels': {'a': 1, 'b': 2},
        },
    }
    old_patches = list(diff(OLD_DOC, edited_old, dot_notation=False))
    edited_new = migrate_document(edited_old, MAPPING)
    base_new = migrate_document(OLD_DOC, MAPPING)

    migrated = migrate_patches(old_patches, MAPPING)
    swapped_then_migrated = migrate_patches(
        list(swap(old_patches)), MAPPING)

    assert patch(swapped_then_migrated, edited_new) == base_new
    assert revert(migrated, edited_new) == base_new
    assert swapped_then_migrated == list(swap(migrated))


def test_integer_indices_keep_their_type():
    old = {'items': [{'v': 1}, {'v': 2}]}
    new = {'items': [{'v': 1}, {'v': 2}, {'v': 3}]}
    old_patches = list(diff(old, new, dot_notation=False))

    migrated = migrate_patches(old_patches, {('items',): ('things',)})

    assert migrated == [
        ('add', ['things'], [(2, {'v': 3})]),
    ]
    assert isinstance(migrated[0][2][0][0], int)


def test_relative_add_remove_keys_are_rewritten_and_regrouped():
    patches = [
        ('add', '', [('user', {'name': 'Ada', 'email': 'e@x'}),
                     ('other', 1)]),
        ('remove', ['user'], [('email', 'ada@example.com')]),
    ]
    mapping = {
        ('user',): ('account',),
        ('user', 'name'): ('account', 'display_name'),
        ('user', 'email'): ('account', 'contact_email'),
    }

    assert migrate_patches(patches, mapping) == [
        ('add', '', [('account', {'name': 'Ada', 'email': 'e@x'}),
                     ('other', 1)]),
        ('remove', ['account'], [('contact_email', 'ada@example.com')]),
    ]


def test_values_named_like_fields_are_not_touched():
    patches = [
        ('change', ['user', 'name'], ('user', 'account')),
        ('add', '', [('user', 'user')]),
    ]
    migrated = migrate_patches(patches, {('user',): ('account',)})

    assert migrated == [
        ('change', ['account', 'name'], ('user', 'account')),
        ('add', '', [('account', 'user')]),
    ]


def test_dotted_string_nodes_are_accepted():
    patches = [('change', 'user.name', ('Ada', 'Grace'))]
    migrated = migrate_patches(patches, MAPPING)

    assert migrated == [
        ('change', ['account', 'display_name'], ('Ada', 'Grace'))]


def test_most_specific_prefix_wins_over_parent_mapping():
    mapping = {
        ('a',): ('x',),
        ('a', 'keep'): ('a', 'keep'),
    }
    patches = [
        ('change', ['a', 'keep'], (1, 2)),
        ('change', ['a', 'moved'], (3, 4)),
    ]

    assert migrate_patches(patches, mapping) == [
        ('change', ['a', 'keep'], (1, 2)),
        ('change', ['x', 'moved'], (3, 4)),
    ]


def test_simultaneous_swap_of_two_names():
    mapping = {('a',): ('b',), ('b',): ('a',)}
    patches = [
        ('change', ['a', 'x'], (1, 2)),
        ('change', ['b', 'x'], (3, 4)),
    ]

    assert migrate_patches(patches, mapping) == [
        ('change', ['b', 'x'], (1, 2)),
        ('change', ['a', 'x'], (3, 4)),
    ]


def test_swap_is_idempotent_through_two_rounds():
    mapping = {('a',): ('b',), ('b',): ('a',)}
    once = migrate_patches([('change', ['a'], (1, 2))], mapping)
    twice = migrate_patches(once, mapping)

    assert twice == [('change', ['a'], (1, 2))]


def test_target_collision_is_diagnosed_before_patching():
    mapping = {('a',): ('x',), ('b',): ('x',)}
    with pytest.raises(MigrationError) as exc:
        migrate_patches([('change', ['a'], (1, 2))], mapping)
    assert exc.value.code == 'target_collision'
    assert exc.value.paths == (('a',), ('b',))


def test_runtime_target_collision_is_diagnosed():
    mapping = {('a', 'x'): ('z',), ('b', 'x'): ('z',)}
    patches = [
        ('change', ['a', 'x'], (1, 2)),
        ('change', ['b', 'x'], (3, 4)),
    ]
    with pytest.raises(MigrationError) as exc:
        migrate_patches(patches, mapping)
    assert exc.value.code == 'target_collision'


def test_pure_swap_cycle_is_allowed_but_longer_cycle_is_diagnosed():
    safe = {('a',): ('b',), ('b',): ('a',)}
    assert migrate_patches(
        [('change', ['a'], (1, 2))], safe) == [
        ('change', ['b'], (1, 2))]

    cyclic = {('a',): ('b',), ('b',): ('c',), ('c',): ('a',)}
    with pytest.raises(MigrationError) as exc:
        migrate_patches([('change', ['a'], (1, 2))], cyclic)
    assert exc.value.code == 'mapping_cycle'


def test_lost_context_for_change_above_mapped_field():
    mapping = {('a', 'x'): ('b', 'x')}
    with pytest.raises(MigrationError) as exc:
        migrate_patches([('change', ['a'], ({'x': 1}, {'x': 2}))], mapping)
    assert exc.value.code == 'lost_context'


def test_lost_context_for_add_remove_descending_into_mapped_field():
    mapping = {('a', 'x'): ('b', 'x')}
    with pytest.raises(MigrationError) as exc:
        migrate_patches([('add', '', [('a', {'y': 1})])], mapping)
    assert exc.value.code == 'lost_context'


def test_whole_subtree_add_only_rewrites_its_own_parent_path():
    mapping = {('a',): ('b',)}
    assert migrate_patches(
        [('add', '', [('a', {'x': 1})])], mapping) == [
        ('add', '', [('b', {'x': 1})])]


def test_invalid_segment_type_is_diagnosed():
    with pytest.raises(MigrationError) as exc:
        migrate_patches(
            [('change', ['a'], (1, 2))], {('a',): (None,)})
    assert exc.value.code == 'invalid_mapping'


def test_duplicate_old_path_is_diagnosed():
    mapping = [(('a',), ('x',)), (('a',), ('y',))]
    with pytest.raises(MigrationError) as exc:
        migrate_patches([('change', ['a'], (1, 2))], mapping)
    assert exc.value.code == 'invalid_mapping'


def test_migrate_single_patch_helper():
    patch_item = ('change', ['a'], (1, 2))
    assert migrate_patch(patch_item, {('a',): ('b',)}) == (
        'change', ['b'], (1, 2))


def test_mapping_iterable_of_pairs_is_accepted():
    assert migrate_patches(
        [('change', ['a'], (1, 2))], [(('a',), ('b',))]) == [
        ('change', ['b'], (1, 2))]


def test_migrate_document_with_integer_segments():
    document = {'rows': [{'v': 1}, {'v': 2}]}
    assert migrate_document(document, {('rows',): ('items',)}) == {
        'items': [{'v': 1}, {'v': 2}]}


def test_set_patches_keep_integer_relative_index():
    old = {'groups': {'a': {1, 2}}}
    new = {'groups': {'a': {1, 2, 3}}}
    patches = list(diff(old, new, dot_notation=False))

    migrated = migrate_patches(
        patches, {('groups', 'a'): ('sets', 'alpha')})

    assert migrated == [('add', ['sets', 'alpha'], [(0, {3})])]
    assert isinstance(migrated[0][2][0][0], int)
    assert patch(migrated, migrate_document(old, {
        ('groups', 'a'): ('sets', 'alpha')})) == migrate_document(
            new, {('groups', 'a'): ('sets', 'alpha')})
