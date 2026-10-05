# SPDX-FileCopyrightText: 2026 CERN.
# SPDX-License-Identifier: MIT

import random

from dictdiffer import diff, patch, revert
from dictdiffer.utils import RecordMatcher


def _pair_cost(first_record, second_record):
    fields = set(first_record) | set(second_record)
    return sum(1 for field in fields
               if first_record.get(field) != second_record.get(field))


def _brute_force_cost(first, second):
    best = [(-1, 0)]  # (number of pairs, cost)

    def search(index, used, count, cost):
        if index == len(first):
            if (count, -cost) > (best[0][0], -best[0][1]):
                best[0] = (count, cost)
            return
        search(index + 1, used, count, cost)
        for other in range(len(second)):
            if other in used:
                continue
            if first[index].get('id') != second[other].get('id'):
                continue
            search(index + 1, used | {other}, count + 1,
                   cost + _pair_cost(first[index], second[other]))

    search(0, frozenset(), 0, 0)
    return best[0][1]


def _random_record(rng):
    record = {'id': rng.randrange(3)}
    if rng.random() < 0.8:
        record['a'] = rng.randrange(3)
    if rng.random() < 0.8:
        record['b'] = rng.randrange(3)
    return record


def test_match_pairs_by_identity():
    matcher = RecordMatcher('id')
    alignment = matcher.match(
        [{'id': 1, 'v': 'a'}, {'id': 2, 'v': 'b'}],
        [{'id': 2, 'v': 'b'}, {'id': 1, 'v': 'a'}])
    assert alignment.pairs == ((0, 1), (1, 0))
    assert alignment.mapping == {0: 1, 1: 0}
    assert alignment.removed == ()
    assert alignment.added == ()


def test_match_minimum_cost_with_duplicate_identities():
    matcher = RecordMatcher('id')
    alignment = matcher.match(
        [{'id': 1, 'v': 'a'}, {'id': 1, 'v': 'b'}],
        [{'id': 1, 'v': 'b'}, {'id': 1, 'v': 'c'}])
    assert alignment.pairs == ((0, 1), (1, 0))


def test_match_tie_break_is_deterministic():
    matcher = RecordMatcher('id')
    first = [{'id': 1, 'v': 'a'}, {'id': 1, 'v': 'b'}]
    second = [{'id': 1, 'v': 'x'}, {'id': 1, 'v': 'y'}]
    alignment = matcher.match(first, second)
    assert alignment.pairs == ((0, 0), (1, 1))
    assert matcher.match(first, second) == alignment
    assert matcher.match(list(first), list(second)) == alignment


def test_match_respects_field_cost():
    first = [{'id': 1, 'x': 0, 'y': 0}, {'id': 1, 'x': 1, 'y': 1}]
    second = [{'id': 1, 'x': 0, 'y': 1}, {'id': 1, 'x': 1, 'y': 0}]
    # uniform cost ties, the lexicographically smallest pairing wins
    assert RecordMatcher('id').match(first, second).pairs == ((0, 0), (1, 1))
    # an expensive 'y' field flips the optimal pairing
    matcher = RecordMatcher('id', field_cost={'y': 5.0})
    assert matcher.match(first, second).pairs == ((0, 1), (1, 0))
    # a callable cost gives the same result
    matcher = RecordMatcher(
        'id',
        field_cost=lambda field, old, new: 5.0 if field == 'y' else 1.0)
    assert matcher.match(first, second).pairs == ((0, 1), (1, 0))


def test_match_cost_matches_exhaustive_brute_force():
    rng = random.Random(20261005)
    for _ in range(200):
        first = [_random_record(rng) for _ in range(rng.randrange(5))]
        second = [_random_record(rng) for _ in range(rng.randrange(5))]
        alignment = RecordMatcher('id').match(first, second)
        cost = sum(_pair_cost(first[i], second[j])
                   for i, j in alignment.pairs)
        assert cost == _brute_force_cost(first, second)
        # the pairing is one to one and respects the identities
        assert len(set(i for i, _ in alignment.pairs)) == len(alignment.pairs)
        assert len(set(j for _, j in alignment.pairs)) == len(alignment.pairs)
        for i, j in alignment.pairs:
            assert first[i]['id'] == second[j]['id']
        matched_first = set(i for i, _ in alignment.pairs)
        matched_second = set(j for _, j in alignment.pairs)
        assert sorted(matched_first | set(alignment.removed)) == \
            list(range(len(first)))
        assert sorted(matched_second | set(alignment.added)) == \
            list(range(len(second)))


def test_match_greedy_fallback_is_deterministic():
    first = [{'id': 1, 'v': i} for i in range(10)]
    second = [{'id': 1, 'v': i} for i in reversed(range(10))]
    matcher = RecordMatcher('id', exhaustive_limit=2)
    alignment = matcher.match(first, second)
    assert alignment == matcher.match(first, second)
    assert all(first[i] == second[j] for i, j in alignment.pairs)


def test_diff_insertion():
    first = [{'id': 1, 'v': 1}, {'id': 2, 'v': 2}]
    second = [{'id': 0, 'v': 0}] + [dict(record) for record in first]
    matcher = RecordMatcher('id')
    result = list(diff(first, second, record_matcher=matcher))
    assert result == [('add', '', [(0, {'id': 0, 'v': 0})])]
    assert patch(result, first) == second
    assert revert(result, second) == first
    assert matcher.mappings == {(): {0: 1, 1: 2}}


def test_diff_removal():
    first = [{'id': 1}, {'id': 2}, {'id': 3}]
    second = [{'id': 1}, {'id': 3}]
    matcher = RecordMatcher('id')
    result = list(diff(first, second, record_matcher=matcher))
    assert result == [('remove', '', [(1, {'id': 2})])]
    assert patch(result, first) == second
    assert revert(result, second) == first
    assert matcher.mappings == {(): {0: 0, 2: 1}}


def test_diff_reorder():
    first = [{'id': 1}, {'id': 2}, {'id': 3}]
    second = [{'id': 3}, {'id': 1}, {'id': 2}]
    matcher = RecordMatcher('id')
    result = list(diff(first, second, record_matcher=matcher))
    assert result == [
        ('remove', '', [(2, {'id': 3})]),
        ('add', '', [(0, {'id': 3})]),
    ]
    assert patch(result, first) == second
    assert matcher.mappings == {(): {0: 1, 1: 2, 2: 0}}


def test_diff_duplicate_identity_records_are_not_merged():
    first = [{'id': 1, 'v': 'x'}]
    second = [{'id': 1, 'v': 'x'}, {'id': 1, 'v': 'x'}]
    matcher = RecordMatcher('id')
    result = list(diff(first, second, record_matcher=matcher))
    assert result == [('add', '', [(1, {'id': 1, 'v': 'x'})])]
    patched = patch(result, first)
    assert patched == second
    assert len(patched) == 2
    assert revert(result, second) == first
    assert matcher.mappings == {(): {0: 0}}

    result = list(diff(second, first, record_matcher=matcher))
    assert result == [('remove', '', [(1, {'id': 1, 'v': 'x'})])]
    assert patch(result, second) == first
    assert revert(result, first) == second


def test_diff_modified_record():
    first = [{'id': 1, 'v': 1}, {'id': 2, 'v': 2}]
    second = [{'id': 1, 'v': 1}, {'id': 2, 'v': 3}]
    matcher = RecordMatcher('id')
    result = list(diff(first, second, record_matcher=matcher))
    assert result == [('change', [1, 'v'], (2, 3))]
    assert patch(result, first) == second
    assert revert(result, second) == first
    assert matcher.mappings == {(): {0: 0, 1: 1}}


def test_diff_insert_reorder_duplicate_and_modification():
    first = [
        {'id': 1, 'v': 'a'},
        {'id': 2, 'v': 'b'},
        {'id': 2, 'v': 'c'},
        {'id': 3, 'v': 'd'},
    ]
    second = [
        {'id': 3, 'v': 'd'},
        {'id': 2, 'v': 'c'},
        {'id': 2, 'v': 'B'},
        {'id': 4, 'v': 'e'},
    ]
    matcher = RecordMatcher('id')
    result = list(diff(first, second, record_matcher=matcher))
    assert ('change', [1, 'v'], ('b', 'B')) in result
    assert ('remove', '', [(0, {'id': 1, 'v': 'a'})]) in result
    assert ('add', '', [(3, {'id': 4, 'v': 'e'})]) in result
    assert patch(result, first) == second
    assert matcher.mappings == {(): {1: 2, 2: 1, 3: 0}}


def test_diff_nested_record_lists():
    first = {'records': [{'id': 1}, {'id': 2}]}
    second = {'records': [{'id': 2}, {'id': 1}]}
    matcher = RecordMatcher('id')
    result = list(diff(first, second, record_matcher=matcher))
    assert patch(result, first) == second
    assert matcher.mappings == {('records',): {0: 1, 1: 0}}


def test_diff_record_without_identity_is_added_and_removed():
    first = [{'id': 1}, {'name': 'old'}]
    second = [{'id': 1}, {'name': 'new'}]
    matcher = RecordMatcher('id')
    result = list(diff(first, second, record_matcher=matcher))
    assert result == [
        ('remove', '', [(1, {'name': 'old'})]),
        ('add', '', [(1, {'name': 'new'})]),
    ]
    assert patch(result, first) == second
    assert matcher.mappings == {(): {0: 0}}


def test_diff_empty_record_lists():
    matcher = RecordMatcher('id')
    assert list(diff([], [], record_matcher=matcher)) == []
    result = list(diff([], [{'id': 1}], record_matcher=matcher))
    assert result == [('add', '', [(0, {'id': 1})])]
    assert matcher.mappings == {(): {}}


def test_diff_record_matcher_ignores_non_record_lists():
    matcher = RecordMatcher('id')
    result = list(diff([1, 2], [1, 3], record_matcher=matcher))
    assert result == [('change', [1], (2, 3))]
    assert matcher.mappings == {}


def test_diff_lists_are_positional_by_default():
    first = [{'id': 1}, {'id': 2}]
    second = [{'id': 0}, {'id': 1}, {'id': 2}]
    result = list(diff(first, second))
    assert result == [
        ('change', [0, 'id'], (1, 0)),
        ('change', [1, 'id'], (2, 1)),
        ('add', '', [(2, {'id': 2})]),
    ]
