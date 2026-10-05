# SPDX-FileCopyrightText: 2026 TU Wien.
# SPDX-License-Identifier: MIT

"""Record alignment strategies for diffing lists of business records."""

from collections.abc import Mapping, MutableMapping, MutableSequence
from itertools import combinations, permutations

MISSING = object()


class RecordAlignment(object):
    """Align lists of records by identity before computing the diff.

    The strategy matches records (dictionaries) of two lists by their
    identity instead of their position, so that one inserted record is
    not reported as a change of every following record.

    The identity does not have to be unique. Records sharing the same
    identity are paired so that the total field difference cost is
    minimal, ties are resolved deterministically and records that cannot
    be paired are reported as additions or removals.

        >>> from dictdiffer import diff
        >>> from dictdiffer.align import RecordAlignment
        >>> first = [{'id': 1, 'v': 'a'}, {'id': 2, 'v': 'b'}]
        >>> second = [{'id': 2, 'v': 'b'}, {'id': 1, 'v': 'a'}]
        >>> result = diff(first, second, align=RecordAlignment('id'))
        >>> list(result)
        [('remove', '', [(0, {'id': 1, 'v': 'a'})]), ('add', '', [(1, {'id': 1, 'v': 'a'})])]
        >>> result.mapping
        {0: 1, 1: 0}
    """

    def __init__(self, identity, field_cost=None):
        """Initialize the record alignment strategy.

        :param identity: Field name, iterable of field names or callable
                         returning the identity key of a record.
        :param field_cost: Cost of a differing field. It can be a mapping
                           of field names to weights (fields missing from
                           the mapping have weight 1) or a callable
                           ``field_cost(field, old_value, new_value)``.
                           Missing values are passed as
                           ``dictdiffer.align.MISSING``. By default every
                           differing field costs 1.
        """
        if isinstance(identity, str):
            self._identity_fields = (identity, )
            self._identity_key = None
        elif callable(identity):
            self._identity_fields = None
            self._identity_key = identity
        else:
            self._identity_fields = tuple(identity)
            self._identity_key = None
        self.identity = identity
        self.field_cost = field_cost

    def key(self, record):
        """Return the identity key of a record."""
        if self._identity_key is not None:
            return self._identity_key(record)
        if len(self._identity_fields) == 1:
            return record[self._identity_fields[0]]
        return tuple(record[field] for field in self._identity_fields)

    def applies(self, first, second):
        """Check whether the alignment can be applied to the two lists.

        Both objects have to be mutable sequences containing only
        dictionaries that provide the identity field(s).
        """
        if not (isinstance(first, MutableSequence) and
                isinstance(second, MutableSequence)):
            return False
        for record in list(first) + list(second):
            if not isinstance(record, MutableMapping):
                return False
            if self._identity_fields is not None:
                if not all(field in record
                           for field in self._identity_fields):
                    return False
            else:
                try:
                    hash(self._identity_key(record))
                except Exception:
                    return False
        return True

    def field_difference_cost(self, field, old, new):
        """Return the cost of one differing field of a record."""
        if self.field_cost is None:
            return 0 if old == new else 1
        if isinstance(self.field_cost, Mapping):
            return 0 if old == new else self.field_cost.get(field, 1)
        return self.field_cost(field, old, new)

    def pair_cost(self, first_record, second_record):
        """Return the total cost of pairing two records.

        The cost is the sum of the field difference costs over all fields
        except the identity field(s).

            >>> RecordAlignment('id').pair_cost(
            ...     {'id': 1, 'v': 'a'}, {'id': 1, 'v': 'b'})
            1
            >>> RecordAlignment('id').pair_cost(
            ...     {'id': 1, 'v': 'a'}, {'id': 1, 'v': 'a'})
            0
        """
        identity_fields = self._identity_fields or ()
        return sum(
            self.field_difference_cost(
                field,
                first_record.get(field, MISSING),
                second_record.get(field, MISSING))
            for field in set(first_record) | set(second_record)
            if field not in identity_fields
        )

    def align(self, first, second):
        """Match the records of two lists minimizing the pairing cost.

        Return a tuple ``(pairs, removed, added)`` where ``pairs`` is a
        list of ``(first_index, second_index)`` tuples sorted by
        ``first_index``, ``removed`` are the sorted indices of unmatched
        records in ``first`` and ``added`` the sorted indices of
        unmatched records in ``second``.

            >>> RecordAlignment('id').align(
            ...     [{'id': 1}, {'id': 1}], [{'id': 1}])
            ([(0, 0)], [1], [])
        """
        groups = {}
        for index, record in enumerate(first):
            groups.setdefault(self.key(record), [[], []])[0].append(index)
        for index, record in enumerate(second):
            groups.setdefault(self.key(record), [[], []])[1].append(index)

        pairs, removed, added = [], [], []
        for first_indices, second_indices in groups.values():
            group_pairs, group_removed, group_added = self._match_group(
                first_indices, second_indices, first, second)
            pairs.extend(group_pairs)
            removed.extend(group_removed)
            added.extend(group_added)

        pairs.sort()
        removed.sort()
        added.sort()
        return pairs, removed, added

    def _match_group(self, first_indices, second_indices, first, second):
        """Find the minimum cost pairing inside one identity group.

        All pairings are enumerated, hence the result is guaranteed to be
        optimal and deterministic: among pairings with equal minimal cost
        the lexicographically smallest one is selected.
        """
        size = min(len(first_indices), len(second_indices))
        if size == 0:
            return [], list(first_indices), list(second_indices)

        costs = {}
        for first_index in first_indices:
            for second_index in second_indices:
                costs[(first_index, second_index)] = self.pair_cost(
                    first[first_index], second[second_index])

        best = None
        for first_combo in combinations(first_indices, size):
            for second_perm in permutations(second_indices, size):
                candidate = tuple(zip(first_combo, second_perm))
                total = sum(costs[pair] for pair in candidate)
                if best is None or (total, candidate) < best:
                    best = (total, candidate)

        pairs = list(best[1])
        matched_first = set(index for index, _ in pairs)
        matched_second = set(index for _, index in pairs)
        removed = [i for i in first_indices if i not in matched_first]
        added = [j for j in second_indices if j not in matched_second]
        return pairs, removed, added


def _anchored_pairs(pairs):
    """Split matched pairs into anchored pairs and moved pairs.

    Anchored pairs form the longest subsequence of pairs (sorted by
    ``first_index``) whose ``second_index`` values are increasing, hence
    they keep their relative order and do not need to be moved. The
    longest subsequence is reconstructed deterministically.
    """
    if not pairs:
        return [], []

    second_indices = [second_index for _, second_index in pairs]
    lengths = [1] * len(pairs)
    for current in range(len(pairs)):
        for previous in range(current):
            if (second_indices[previous] < second_indices[current] and
                    lengths[previous] + 1 > lengths[current]):
                lengths[current] = lengths[previous] + 1

    anchored = []
    needed = max(lengths)
    for current in range(len(pairs) - 1, -1, -1):
        if lengths[current] == needed and (
                not anchored or
                second_indices[current] < second_indices[anchored[0]]):
            anchored.insert(0, current)
            needed -= 1

    anchored = set(anchored)
    return (
        [pair for index, pair in enumerate(pairs) if index in anchored],
        [pair for index, pair in enumerate(pairs) if index not in anchored],
    )


class AlignedDiff(list):
    """List of diff tuples annotated with record alignment information.

    The object behaves like a regular list of diff tuples, hence it can
    be passed to :func:`dictdiffer.patch`, :func:`dictdiffer.swap` and
    :func:`dictdiffer.revert`. In addition, the ``mappings`` attribute
    describes how the record positions changed between the two lists.
    """

    def __init__(self, items=(), mappings=None):
        """Initialize the diff result.

        :param items: Iterable of diff tuples.
        :param mappings: Dictionary mapping a node path (tuple) to a
                         dictionary of original to new record positions.
        """
        super(AlignedDiff, self).__init__(items)
        self.mappings = dict(mappings) if mappings else {}

    @property
    def mapping(self):
        """Return the position mapping of the top-level list of records."""
        return self.mappings.get((), {})
