# SPDX-FileCopyrightText: 2015 CERN.
# SPDX-FileCopyrightText: 2017, 2019 ETH Zurich, Swiss Data Science Center, Jiri Kuncar.
# SPDX-License-Identifier: MIT

"""Utils gathers helper functions, classes for the dictdiffer module."""

import math
import sys
from decimal import Decimal
from itertools import zip_longest

num_types = int, float, Decimal
EPSILON = sys.float_info.epsilon


class NumericTolerance(object):
    """Numeric comparison strategy for a set of structured paths.

    Describes when two numbers are considered equal:

    - ``abs_tol``: maximum allowed absolute difference.
    - ``rel_tol``: maximum allowed difference relative to the largest
      absolute value of the two operands.
    - ``nan_equal``: whether two NaN values are considered equal.

    Applies to ``int``, ``float`` and :class:`decimal.Decimal` values.
    ``bool`` is never treated as a number. Signed zeros compare equal
    (``0.0 == -0.0``). Infinities are only equal to an infinity of the
    same sign; an infinity is never within tolerance of a finite value.
    """

    def __init__(self, abs_tol=None, rel_tol=None, nan_equal=True):
        """Initialize the numeric tolerance strategy.

        :param abs_tol: Absolute tolerance, a non-negative number or None.
        :param rel_tol: Relative tolerance, a non-negative number or None.
        :param nan_equal: Consider two NaN values as equal.
        """
        for name, value in (('abs_tol', abs_tol), ('rel_tol', rel_tol)):
            if value is not None:
                if isinstance(value, bool) or \
                        not isinstance(value, num_types):
                    raise TypeError(
                        '{0} must be a real number or None'.format(name))
                if value != value:
                    raise ValueError('{0} must not be NaN'.format(name))
                if value < 0:
                    raise ValueError('{0} must not be negative'.format(name))
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.nan_equal = bool(nan_equal)

    def __repr__(self):
        """Return string representation."""
        return 'NumericTolerance(abs_tol={0!r}, rel_tol={1!r}, ' \
               'nan_equal={2!r})'.format(self.abs_tol, self.rel_tol,
                                         self.nan_equal)


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


class ToleranceMap(WildcardDict):
    """Map structured paths to :class:`NumericTolerance` strategies.

    Keys are path tuples like ``('sensors', '+', 'value')`` supporting the
    ``+`` (single level) and ``*`` (everything that follows) wildcards.
    Lookup follows a deterministic priority: exact path match first, then
    ``+`` wildcard, then the longest matching ``*`` prefix.

    Validation only inspects the configured paths and strategies, never
    the diffed business objects.
    """

    def __init__(self, values=None):
        """Initialize and validate the path to strategy mapping.

        :param values: a dictionary mapping path tuples to
                       :class:`NumericTolerance` instances (or dicts of
                       keyword arguments for it).
        """
        if values is not None:
            converted = {}
            for key, value in values.items():
                if not isinstance(key, tuple) or not key:
                    raise TypeError(
                        'tolerance paths must be non-empty tuples, '
                        'got {0!r}'.format(key))
                if not isinstance(value, NumericTolerance):
                    if isinstance(value, dict):
                        value = NumericTolerance(**value)
                    else:
                        raise TypeError(
                            'tolerance strategies must be NumericTolerance '
                            'instances, got {0!r}'.format(value))
                converted[key] = value
            values = converted
        super(ToleranceMap, self).__init__(values)

    def query(self, key):
        """Return the strategy for the path or None if nothing matches."""
        try:
            return self[key]
        except KeyError:
            return None


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


def is_number(value):
    """Check if a value is a number (``int``, ``float`` or ``Decimal``).

    ``bool`` values are excluded: ``True`` must not be treated as the
    number ``1``.
    """
    return isinstance(value, num_types) and not isinstance(value, bool)


def _is_nan(value):
    if isinstance(value, Decimal):
        return value.is_nan()
    return isinstance(value, float) and math.isnan(value)


def _is_infinite(value):
    if isinstance(value, Decimal):
        return value.is_infinite()
    return isinstance(value, float) and math.isinf(value)


def numeric_equal(first, second, numeric_tolerance=None):
    """Check if two numbers are equal according to a tolerance strategy.

    Both values must pass :func:`is_number`. The rules are stated
    explicitly so that special values behave the same across the diff and
    the merge conflict detection:

    - Signed zeros (``0.0`` and ``-0.0``) compare equal.
    - Two NaN values compare equal when ``numeric_tolerance.nan_equal``
      (the default); otherwise they compare different. NaN versus any
      other value is always different.
    - Two infinities compare equal only when they share a sign.
    - An infinity can never be within tolerance of a finite value.
    - Otherwise the values are close when
      ``|first - second| <= max(abs_tol,
      rel_tol * max(|first|, |second|))``.
    """
    abs_tol = numeric_tolerance.abs_tol \
        if numeric_tolerance is not None else None
    rel_tol = numeric_tolerance.rel_tol \
        if numeric_tolerance is not None else None

    first_nan, second_nan = _is_nan(first), _is_nan(second)
    if first_nan or second_nan:
        if not (first_nan and second_nan):
            return False
        if numeric_tolerance is None:
            return True
        return numeric_tolerance.nan_equal

    if first == second:
        # signed zeros and same-sign infinities land here
        return True

    first_inf, second_inf = _is_infinite(first), _is_infinite(second)
    if first_inf or second_inf:
        # mixed infinity/finite or opposite signs: an infinity can never
        # be within tolerance of another value
        return False

    absolute_difference = abs(first - second)
    abs_tol = abs_tol if abs_tol is not None else 0
    rel_tol = rel_tol if rel_tol is not None else 0

    if isinstance(first, Decimal) or isinstance(second, Decimal):
        # Decimal and float cannot be combined arithmetically
        abs_tol = Decimal(abs_tol)
        rel_tol = Decimal(rel_tol)

    tolerance = max(abs_tol, rel_tol * max(abs(first), abs(second)))
    return absolute_difference <= tolerance


def are_different(first, second, tolerance, absolute_tolerance=None,
                  numeric_tolerance=None):
    """Check if 2 values are different.

    In case of numerical values, the tolerance is used to check if the values
    are different.
    In all other cases, the difference is straight forward.
    """
    # bool is never treated as a number: True must not compare equal to 1
    if isinstance(first, bool) or isinstance(second, bool):
        return type(first) is not type(second) or first != second

    if first == second:
        # values are same - simple case
        return False

    if numeric_tolerance is not None:
        if is_number(first) and is_number(second):
            return not numeric_equal(first, second, numeric_tolerance)
        # non-numeric values only match when they are equal
        return True

    first_is_nan, second_is_nan = _is_nan(first), _is_nan(second)

    if first_is_nan or second_is_nan:
        # two 'NaN' values are not different (see issue #114)
        return not (first_is_nan and second_is_nan)
    elif is_number(first) and is_number(second):
        # two numerical values are compared with tolerance
        return not numeric_equal(
            first,
            second,
            NumericTolerance(
                abs_tol=absolute_tolerance,
                rel_tol=tolerance,
            ),
        )
    # we got different values
    return True
