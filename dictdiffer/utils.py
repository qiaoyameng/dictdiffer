# SPDX-FileCopyrightText: 2015 CERN.
# SPDX-FileCopyrightText: 2017, 2019 ETH Zurich, Swiss Data Science Center, Jiri Kuncar.
# SPDX-License-Identifier: MIT

"""Utils gathers helper functions, classes for the dictdiffer module."""

import math
import sys
from collections import namedtuple
from itertools import permutations, zip_longest

num_types = int, float
EPSILON = sys.float_info.epsilon


class WildcardDict(dict):
    """Provide possibility to use special wildcard keys to access values.

    Those wildcards are:
        *:  wildcard for everything that follows
        +:  wildcard for anything on the same path level
    The intended use case of this are dictionaries, that utilize tuples as
    keys.

        >>> from dictdiffer.utils import WildcardDict
        >>> w = WildcardDict({('foo', '*'): '* card',
        ...                   ('banana', '+'): '+ card'})
        >>> w[ ('foo', 'bar', 'baz') ]
        '* card'
        >>> w[ ('banana', 'apple') ]
        '+ card'
    """

    def __init__(self, values=None):
        """Set lookup key indices.

        :param values: a dictionary
        """
        super(WildcardDict, self).__init__()
        self.star_keys = set()
        self.plus_keys = set()

        if values is not None:
            for key, value in values.items():
                self.__setitem__(key, value)

    def __getitem__(self, key):
        """Return the value corresponding to the key, regarding wildcards.

        If the key doesn't exit it tries the '+' wildcard and then the
        '*' wildcard.

            >>> w = WildcardDict({('foo', '*'): '* card',
            ...                   ('banana', '+'): '+ card'})
            >>> w[ ('foo', 'bar') ]
            '* card'
            >>> w[ ('foo', 'bar', 'baz') ]
            '* card'
            >>> w[ ('banana', 'apple') ]
            '+ card'
            >>> w[ ('banana', 'apple', 'mango') ]
            Traceback (most recent call last):
                ...
            KeyError
        """
        try:
            return super(WildcardDict, self).__getitem__(key)
        except KeyError:
            if key[:-1] in self.plus_keys:
                return super(WildcardDict, self).__getitem__(key[:-1]+('+',))
            for _key in [key[:-i] for i in range(1, len(key)+1)]:
                if _key in self.star_keys:
                    return super(WildcardDict, self).__getitem__(_key+('*',))
            raise KeyError

    def __setitem__(self, key, value):
        """Set the item for a given key (path)."""
        super(WildcardDict, self).__setitem__(key, value)

        if key[-1] == '+':
            self.plus_keys.add(key[:-1])
        if key[-1] == '*':
            self.star_keys.add(key[:-1])

    def query_path(self, key):
        """Return the key (path) that matches the queried key.

        >>> w = WildcardDict({('foo', '*'): 'banana'})
        >>> w.query_path(('foo', 'bar', 'baz'))
        ('foo', '*')
        """
        if key in self:
            return key
        if key[:-1] in self.plus_keys:
            return key[:-1]+('+',)
        for _key in [key[:-i] for i in range(1, len(key)+1)]:
            if _key in self.star_keys:
                return _key+('*',)

        raise KeyError


class PathLimit(object):
    """Class to limit recursion depth during the dictdiffer.diff execution."""

    def __init__(self, path_limits=[], final_key=None):
        """Initialize a dictionary structure to determine a path limit.

        :param path_limits: list of keys (tuples) determining the path limits
        :param final_key: the key used in the dictionary to determin if the
                          path is final

            >>> pl = PathLimit( [('foo', 'bar')] , final_key='!@#$%FINAL')
            >>> pl.dict
            {'foo': {'bar': {'!@#$%FINAL': True}}}
        """
        self.final_key = final_key if final_key else '!@#$FINAL'
        self.dict = {}
        for key_path in path_limits:
            containing = self.dict
            for key in key_path:
                try:
                    containing = containing[key]
                except KeyError:
                    containing[key] = {}
                    containing = containing[key]

            containing[self.final_key] = True

    def path_is_limit(self, key_path):
        """Query the PathLimit object if the given key_path is a limit.

        >>> pl = PathLimit( [('foo', 'bar')] , final_key='!@#$%FINAL')
        >>> pl.path_is_limit( ('foo', 'bar') )
        True
        """
        containing = self.dict
        for key in key_path:
            try:
                containing = containing[key]
            except KeyError:
                try:
                    containing = containing['*']
                except KeyError:
                    return False

        return containing.get(self.final_key, False)


def create_dotted_node(node):
    """Create the *dotted node* notation for the dictdiffer.diff patches.

    >>> create_dotted_node( ['foo', 'bar', 'baz'] )
    'foo.bar.baz'
    """
    if all(map(lambda x: isinstance(x, str), node)):
        return '.'.join(node)
    else:
        return list(node)


def get_path(patch):
    """Return the path for a given dictdiffer.diff patch."""
    if patch[1] != '':
        keys = (patch[1].split('.') if isinstance(patch[1], str)
                else patch[1])
    else:
        keys = []
    keys = keys + [patch[2][0][0]] if patch[0] != 'change' else keys
    return tuple(keys)


def is_super_path(path1, path2):
    """Check if one path is the super path of the other.

    Super path means, that the n values in tuple are equal to the first n of m
    vales in tuple b.

        >>> is_super_path( ('foo', 'bar'), ('foo', 'bar') )
        True

        >>> is_super_path( ('foo', 'bar'), ('foo', 'bar', 'baz') )
        True

        >>> is_super_path( ('foo', 'bar'), ('foo', 'apple', 'banana') )
        False
    """
    return all(map(lambda x: x[0] == x[1] or x[0] is None,
                   zip_longest(path1, path2)))


def nested_hash(obj):
    """Create a hash of nested, mutable data structures.

    It shall be noted, that the uniqeness of those hashes in general cases is
    not assured but it should be enough for the cases occurring during the
    merging process.
    """
    try:
        return hash(obj)
    except TypeError:
        if isinstance(obj, (list, tuple)):
            return hash(tuple(map(nested_hash, obj)))
        elif isinstance(obj, set):
            return hash(tuple(map(nested_hash, sorted(obj))))
        elif isinstance(obj, dict):
            return hash(tuple(map(nested_hash, sorted(obj.items()))))


def dot_lookup(source, lookup, parent=False):
    """Allow you to reach dictionary items with string or list lookup.

    Recursively find value by lookup key split by '.'.

        >>> from dictdiffer.utils import dot_lookup
        >>> dot_lookup({'a': {'b': 'hello'}}, 'a.b')
        'hello'

    If parent argument is True, returns the parent node of matched
    object.

        >>> dot_lookup({'a': {'b': 'hello'}}, 'a.b', parent=True)
        {'b': 'hello'}

    If node is empty value, returns the whole dictionary object.

        >>> dot_lookup({'a': {'b': 'hello'}}, '')
        {'a': {'b': 'hello'}}

    """
    if lookup is None or lookup == '' or lookup == []:
        return source

    value = source
    if isinstance(lookup, str):
        keys = lookup.split('.')
    elif isinstance(lookup, list):
        keys = lookup
    else:
        raise TypeError('lookup must be string or list')

    if parent:
        keys = keys[:-1]

    for key in keys:
        if isinstance(value, list):
            key = int(key)
        value = value[key]
    return value


def are_different(first, second, tolerance, absolute_tolerance=None):
    """Check if 2 values are different.

    In case of numerical values, the tolerance is used to check if the values
    are different.
    In all other cases, the difference is straight forward.
    """
    if first == second:
        # values are same - simple case
        return False

    first_is_nan, second_is_nan = bool(first != first), bool(second != second)

    if first_is_nan or second_is_nan:
        # two 'NaN' values are not different (see issue #114)
        return not (first_is_nan and second_is_nan)
    elif isinstance(first, num_types) and isinstance(second, num_types):
        # two numerical values are compared with tolerance
        return not math.isclose(
            first,
            second,
            rel_tol=tolerance or 0,
            abs_tol=absolute_tolerance or 0,
        )
    # we got different values
    return True


_MISSING = object()


class Alignment(namedtuple('Alignment', ['pairs', 'removed', 'added'])):
    """Describe how the records of two lists correspond to each other.

    ``pairs`` holds ``(old_index, new_index)`` tuples of matched records
    sorted by the old index, ``removed`` the old indices without a match
    and ``added`` the new indices without a match.
    """

    __slots__ = ()

    @property
    def mapping(self):
        """Return the old position to new position mapping of the pairs."""
        return dict(self.pairs)


def _exhaustive_assignment(costs, size_first, size_second):
    """Return the minimum cost index pairs of a cost matrix.

    Every row (or column, when there are fewer columns than rows) is
    matched exactly once.  Ties are broken deterministically by choosing
    the lexicographically smallest tuple of pairs.
    """
    best = None
    if size_first <= size_second:
        for perm in permutations(range(size_second), size_first):
            pairs = tuple(enumerate(perm))
            candidate = (sum(costs[i][j] for i, j in pairs), pairs)
            if best is None or candidate < best:
                best = candidate
    else:
        for perm in permutations(range(size_first), size_second):
            pairs = tuple(sorted((i, j) for j, i in enumerate(perm)))
            candidate = (sum(costs[i][j] for i, j in pairs), pairs)
            if best is None or candidate < best:
                best = candidate
    return best[1]


def _greedy_assignment(costs, size_first, size_second):
    """Return index pairs by repeatedly picking the cheapest remaining one.

    Used as a fallback for groups too large for the exhaustive search.
    The result is deterministic because candidates are considered in
    ``(cost, first_index, second_index)`` order.
    """
    candidates = sorted(
        (costs[i][j], i, j)
        for i in range(size_first) for j in range(size_second)
    )
    remaining_first = set(range(size_first))
    remaining_second = set(range(size_second))
    pairs = []
    for cost, i, j in candidates:
        if i in remaining_first and j in remaining_second:
            pairs.append((i, j))
            remaining_first.discard(i)
            remaining_second.discard(j)
    return tuple(sorted(pairs))


class RecordMatcher(object):
    """Align lists of record dictionaries by identity fields.

    Records sharing the same identity are paired so that the total field
    difference cost of the pairs is minimal.  When several pairings have
    the same minimal cost, the lexicographically smallest tuple of
    ``(old_index, new_index)`` pairs is chosen, so the result is always
    deterministic.  Records without a partner are reported as removals
    and additions.  Pairing is always one to one, therefore two distinct
    record objects with equal values are never merged into one.

        >>> from dictdiffer.utils import RecordMatcher
        >>> matcher = RecordMatcher('id')
        >>> alignment = matcher.match([{'id': 1}, {'id': 2}],
        ...                           [{'id': 2}, {'id': 1}])
        >>> alignment.pairs
        ((0, 1), (1, 0))
        >>> alignment.mapping
        {0: 1, 1: 0}
        >>> alignment = matcher.match([{'id': 1, 'v': 'a'}],
        ...                           [{'id': 1, 'v': 'a'},
        ...                            {'id': 1, 'v': 'a'}])
        >>> alignment
        Alignment(pairs=((0, 0),), removed=(), added=(1,))
    """

    def __init__(self, identity, field_cost=None, default_cost=1.0,
                 exhaustive_limit=8):
        """Initialize the record matcher.

        :param identity: Field name or iterable of field names identifying
                         a record.  Records missing any of the fields are
                         never matched.
        :param field_cost: Mapping of field name to difference cost, or a
                           callable ``(field, old_value, new_value) -> cost``.
                           Fields not present in a mapping are charged
                           ``default_cost``.
        :param default_cost: Cost of a differing field not covered by
                             ``field_cost``.
        :param exhaustive_limit: Maximum group size for which the pairing
                                 is verified by exhaustive search.  Larger
                                 groups fall back to a deterministic greedy
                                 assignment.
        """
        if isinstance(identity, str):
            identity = (identity,)
        self.identity = tuple(identity)
        self.field_cost = field_cost
        self.default_cost = default_cost
        self.exhaustive_limit = exhaustive_limit
        self.mappings = {}

    def _identity_key(self, record):
        """Return the hashable identity key of a record or None."""
        key = []
        for field in self.identity:
            value = record.get(field, _MISSING)
            if value is _MISSING:
                return None
            key.append(value)
        try:
            hash(tuple(key))
        except TypeError:
            return None
        return tuple(key)

    def _field_cost(self, field, first_value, second_value):
        """Return the difference cost of one field."""
        if self.field_cost is None:
            return self.default_cost
        if callable(self.field_cost):
            return self.field_cost(field, first_value, second_value)
        return self.field_cost.get(field, self.default_cost)

    def _pair_cost(self, first_record, second_record):
        """Return the total field difference cost of pairing two records."""
        cost = 0.0
        for field in set(first_record) | set(second_record):
            first_value = first_record.get(field, _MISSING)
            second_value = second_record.get(field, _MISSING)
            if first_value is _MISSING or second_value is _MISSING or \
                    first_value != second_value:
                cost += self._field_cost(field, first_value, second_value)
        return cost

    def _match_group(self, first_records, second_records):
        """Return the minimum cost pairs of one identity group."""
        size_first = len(first_records)
        size_second = len(second_records)
        costs = [[self._pair_cost(first_records[i], second_records[j])
                  for j in range(size_second)] for i in range(size_first)]
        if min(size_first, size_second) <= self.exhaustive_limit:
            return _exhaustive_assignment(costs, size_first, size_second)
        return _greedy_assignment(costs, size_first, size_second)

    def match(self, first, second):
        """Align the records of two lists.

        :param first: List of record dictionaries.
        :param second: List of record dictionaries.
        :returns: An :class:`Alignment` with the matched ``pairs`` and the
                  ``removed`` and ``added`` indices without a match.
        """
        first_groups = {}
        second_groups = {}
        for index, record in enumerate(first):
            key = self._identity_key(record)
            if key is not None:
                first_groups.setdefault(key, []).append(index)
        for index, record in enumerate(second):
            key = self._identity_key(record)
            if key is not None:
                second_groups.setdefault(key, []).append(index)

        pairs = []
        keys = list(first_groups) + \
            [key for key in second_groups if key not in first_groups]
        for key in keys:
            first_indices = first_groups.get(key, [])
            second_indices = second_groups.get(key, [])
            group_pairs = self._match_group(
                [first[i] for i in first_indices],
                [second[j] for j in second_indices])
            pairs.extend((first_indices[i], second_indices[j])
                         for i, j in group_pairs)

        pairs.sort()
        matched_first = set(i for i, _ in pairs)
        matched_second = set(j for _, j in pairs)
        removed = tuple(i for i in range(len(first)) if i not in matched_first)
        added = tuple(j for j in range(len(second)) if j not in matched_second)
        return Alignment(tuple(pairs), removed, added)
