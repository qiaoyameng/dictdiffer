# SPDX-FileCopyrightText: 2026 TU Wien.
# SPDX-License-Identifier: MIT

import random

import pytest

from dictdiffer import diff, patch, revert, swap
from dictdiffer.align import MISSING, AlignedDiff, RecordAlignment


def brute_force_min_cost(first, second, alignment):
    """Independently compute the minimal total pairing cost.

    As many records as possible are paired; among those pairings the one
    with the minimal total cost wins.
    """
    best = [(0, float('inf'))]
    used = [False] * len(second)

    def visit(index, pairs, total):
        if index == len(first):
            best[0] = min(best[0], (-pairs, total))
            return
        visit(index + 1, pairs, total)  # leave first[index] unmatched
        key = alignment.key(first[index])
        for other, record in enumerate(second):
            if not used[other] and alignment.key(record) == key:
                used[other] = True
                visit(index + 1, pairs + 1,
                      total + alignment.pair_cost(first[index], record))
                used[other] = False

    visit(0, 0, 0)
    return -best[0][0], best[0][1]


def random_records(rand, count):
    """Generate random records with duplicate identities and fields."""
    records = []
    for _ in range(count):
        record = {'id': rand.randint(1, 3)}
        if rand.random() < 0.8:
            record['v'] = rand.choice('abc')
        if rand.random() < 0.5:
            record['w'] = rand.randint(1, 3)
        records.append(record)
    return records


def test_pairing_cost_matches_exhaustive_search():
    alignment = RecordAlignment('id')
    rand = random.Random(20261005)
    for _ in range(300):
        first = random_records(rand, rand.randint(0, 4))
        second = random_records(rand, rand.randint(0, 4))

        pairs, removed, added = alignment.align(first, second)

        # matched pairs share the identity and no record is used twice
        first_indices = [i for i, _ in pairs]
        second_indices = [j for _, j in pairs]
        assert len(set(first_indices)) == len(pairs)
        assert len(set(second_indices)) == len(pairs)
        for i, j in pairs:
            assert alignment.key(first[i]) == alignment.key(second[j])

        # unmatched records are exactly the ones not paired
        assert removed == sorted(
            set(range(len(first))) - set(first_indices))
        assert added == sorted(
            set(range(len(second))) - set(second_indices))

        # the pairing is maximal, optimal and deterministic
        max_pairs, min_cost = brute_force_min_cost(first, second, alignment)
        assert len(pairs) == max_pairs
        total = sum(alignment.pair_cost(first[i], second[j])
                    for i, j in pairs)
        assert total == min_cost
        assert alignment.align(first, second) == (pairs, removed, added)


def test_pairing_cost_with_field_weights():
    alignment = RecordAlignment('id', field_cost={'v': 10})
    rand = random.Random(20261006)
    for _ in range(200):
        first = random_records(rand, rand.randint(0, 4))
        second = random_records(rand, rand.randint(0, 4))
        pairs, _, _ = alignment.align(first, second)
        max_pairs, min_cost = brute_force_min_cost(first, second, alignment)
        assert len(pairs) == max_pairs
        total = sum(alignment.pair_cost(first[i], second[j])
                    for i, j in pairs)
        assert total == min_cost


def test_insertion():
    first = [{'id': 1, 'v': 'a'}, {'id': 2, 'v': 'b'}]
    second = [{'id': 1, 'v': 'a'}, {'id': 9, 'v': 'x'}, {'id': 2, 'v': 'b'}]

    result = diff(first, second, align=RecordAlignment('id'))

    assert isinstance(result, AlignedDiff)
    assert list(result) == [('add', '', [(1, {'id': 9, 'v': 'x'})])]
    assert result.mapping == {0: 0, 1: 2}
    assert result.mappings == {(): {0: 0, 1: 2}}
    assert patch(result, first) == second
    assert revert(result, second) == first


def test_removal():
    first = [{'id': 1}, {'id': 2}, {'id': 3}, {'id': 4}]
    second = [{'id': 1}, {'id': 4}]

    result = diff(first, second, align=RecordAlignment('id'))

    assert list(result) == [
        ('remove', '', [(2, {'id': 3}), (1, {'id': 2})]),
    ]
    assert result.mapping == {0: 0, 3: 1}
    assert patch(result, first) == second
    assert revert(result, second) == first


def test_reorder():
    first = [{'id': 1, 'v': 'a'}, {'id': 2, 'v': 'b'}, {'id': 3, 'v': 'c'}]
    second = [{'id': 3, 'v': 'c'}, {'id': 1, 'v': 'a'}, {'id': 2, 'v': 'b'}]

    result = diff(first, second, align=RecordAlignment('id'))

    assert list(result) == [
        ('remove', '', [(2, {'id': 3, 'v': 'c'})]),
        ('add', '', [(0, {'id': 3, 'v': 'c'})]),
    ]
    assert result.mapping == {0: 1, 1: 2, 2: 0}
    assert patch(result, first) == second


def test_duplicate_identities():
    # identical values in distinct objects are not merged into one
    first = [{'id': 1, 'v': 'a'}, {'id': 1, 'v': 'a'}]
    second = [{'id': 1, 'v': 'a'}]

    result = diff(first, second, align=RecordAlignment('id'))

    assert list(result) == [('remove', '', [(1, {'id': 1, 'v': 'a'})])]
    assert result.mapping == {0: 0}
    assert patch(result, first) == second
    assert revert(result, second) == first

    # nothing changes when both sides have the same duplicates
    result = diff(first, first, align=RecordAlignment('id'))
    assert list(result) == []
    assert result.mapping == {0: 0, 1: 1}

    # the cheapest duplicate is paired
    first = [{'id': 1, 'v': 'a'}, {'id': 1, 'v': 'b'}]
    second = [{'id': 1, 'v': 'b'}]

    result = diff(first, second, align=RecordAlignment('id'))

    assert list(result) == [('remove', '', [(0, {'id': 1, 'v': 'a'})])]
    assert result.mapping == {1: 0}
    assert patch(result, first) == second
    assert revert(result, second) == first


def test_modified_record():
    first = [{'id': 1, 'v': 'a'}, {'id': 2, 'v': 'b'}]
    second = [{'id': 1, 'v': 'a'}, {'id': 2, 'v': 'B'}]

    result = diff(first, second, align=RecordAlignment('id'))

    assert list(result) == [('change', [1, 'v'], ('b', 'B'))]
    assert result.mapping == {0: 0, 1: 1}
    assert patch(result, first) == second
    assert revert(result, second) == first


def test_insertion_reorder_duplicates_and_modification():
    first = [
        {'id': 1, 'v': 'a'},
        {'id': 2, 'v': 'b'},
        {'id': 2, 'v': 'c'},
        {'id': 3, 'v': 'd'},
    ]
    second = [
        {'id': 2, 'v': 'b'},
        {'id': 1, 'v': 'a'},
        {'id': 2, 'v': 'C'},
        {'id': 4, 'v': 'x'},
    ]

    result = diff(first, second, align=RecordAlignment('id'))

    assert list(result) == [
        ('remove', '', [(3, {'id': 3, 'v': 'd'}), (0, {'id': 1, 'v': 'a'})]),
        ('add', '', [(1, {'id': 1, 'v': 'a'}), (3, {'id': 4, 'v': 'x'})]),
        ('change', [2, 'v'], ('c', 'C')),
    ]
    assert result.mapping == {0: 1, 1: 0, 2: 2}
    assert patch(result, first) == second


def test_custom_field_cost_changes_pairing():
    first = [{'id': 1, 'v': 'a', 'w': 1}, {'id': 1, 'v': 'b', 'w': 9}]
    second = [{'id': 1, 'v': 'a', 'w': 9}]

    # with equal weights the lexicographically smallest pairing wins
    result = diff(first, second, align=RecordAlignment('id'))
    assert result.mapping == {0: 0}

    # a heavier 'w' field makes the other pairing cheaper
    result = diff(first, second,
                  align=RecordAlignment('id', field_cost={'w': 10}))
    assert result.mapping == {1: 0}
    assert patch(result, first) == second

    # a callable cost compares the actual values
    alignment = RecordAlignment(
        'id',
        field_cost=lambda field, old, new: 0 if old == new else abs(new - old))
    first = [{'id': 1, 'v': 10}, {'id': 1, 'v': 20}]
    second = [{'id': 1, 'v': 19}]
    result = diff(first, second, align=alignment)
    assert result.mapping == {1: 0}


def test_identity_variants():
    first = [{'id': 1}, {'id': 2}]
    second = [{'id': 2}, {'id': 1}]

    result = diff(first, second, align=RecordAlignment(lambda r: r['id']))
    assert result.mapping == {0: 1, 1: 0}
    assert patch(result, first) == second

    first = [{'t': 1, 'id': 1}, {'t': 1, 'id': 2}]
    second = [{'t': 1, 'id': 2}, {'t': 1, 'id': 1}]

    result = diff(first, second, align=RecordAlignment(('t', 'id')))
    assert result.mapping == {0: 1, 1: 0}
    assert patch(result, first) == second


def test_nested_lists_of_records():
    first = {'items': [{'id': 1}, {'id': 2}]}
    second = {'items': [{'id': 2}, {'id': 1}]}

    result = diff(first, second, align=RecordAlignment('id'))

    assert result.mapping == {}
    assert result.mappings == {('items',): {0: 1, 1: 0}}
    assert patch(result, first) == second


def test_non_records_fall_back_to_positional():
    result = diff([1, 2], [1, 2, 3], align=RecordAlignment('id'))
    assert list(result) == [('add', '', [(2, 3)])]

    result = diff([{'id': 1}, 'not-a-record'], [{'id': 1}],
                  align=RecordAlignment('id'))
    assert list(result) == [('remove', '', [(1, 'not-a-record')])]


def test_positional_mode_is_the_default():
    first = [{'id': 1}, {'id': 2}]
    second = [{'id': 1}, {'id': 9}, {'id': 2}]

    result = diff(first, second)

    assert not isinstance(result, AlignedDiff)
    assert list(result) == [
        ('change', [1, 'id'], (2, 9)),
        ('add', '', [(2, {'id': 2})]),
    ]


def test_expand():
    first = [{'id': 1}, {'id': 2}, {'id': 3}]
    second = [{'id': 2}]

    result = diff(first, second, align=RecordAlignment('id'), expand=True)

    assert list(result) == [
        ('remove', '', [(2, {'id': 3})]),
        ('remove', '', [(0, {'id': 1})]),
    ]
    assert patch(result, first) == second
    assert revert(result, second) == first


def test_immutable_diffs():
    first = [{'id': 1, 'v': 'a'}]
    second = [{'id': 1, 'v': 'a'}, {'id': 2, 'v': 'b'}]

    result = diff(first, second, align=RecordAlignment('id'))
    assert result[0][2][0][1]['v'] == 'b'
    second[1]['v'] = 'changed'
    assert result[0][2][0][1]['v'] == 'b'


def test_swap_reverses_remove_order():
    swapped = list(swap([('remove', '', [(2, 'c'), (1, 'b')])]))
    assert swapped == [('add', '', [(1, 'b'), (2, 'c')])]


def test_missing_field_cost():
    alignment = RecordAlignment('id')
    assert alignment.pair_cost({'id': 1, 'v': 'a'}, {'id': 1}) == 1
    assert alignment.field_difference_cost('v', MISSING, MISSING) == 0
