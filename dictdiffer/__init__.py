# SPDX-FileCopyrightText: 2013 Fatih Erikli.
# SPDX-FileCopyrightText: 2013, 2014, 2015, 2016 CERN.
# SPDX-FileCopyrightText: 2017-2019 ETH Zurich, Swiss Data Science Center, Jiri Kuncar.
# SPDX-License-Identifier: MIT

"""Dictdiffer is a helper module to diff and patch dictionaries."""

from collections.abc import Iterable, MutableMapping, MutableSequence, MutableSet
from copy import deepcopy

from .numeric import (
    IgnoredChange,
    MatchedNumericRule,
    NumericComparison,
    NumericComparisonPolicy,
    NumericPolicy,
    NumericRule,
    NumericTolerance,
    NumericTolerancePolicy,
    PathNumericPolicy,
    PathTolerance,
    TolerancePolicy,
    ToleranceRule,
    ToleranceSpec,
    coerce_numeric_policy,
    is_numeric,
    take_policy_alias,
    validate_numeric_policy,
)
from .utils import EPSILON, PathLimit, dot_lookup
from .version import __version__

(ADD, REMOVE, CHANGE) = (
    'add', 'remove', 'change')

__all__ = (
    'diff', 'patch', 'swap', 'revert', 'dot_lookup', '__version__',
    'DiffResult', 'IgnoredChange', 'MatchedNumericRule',
    'NumericComparison', 'NumericPolicy', 'NumericRule',
    'coerce_numeric_policy',
    'is_numeric',
    'validate_numeric_policy',
)

DICT_TYPES = (MutableMapping, )
LIST_TYPES = (MutableSequence, )
SET_TYPES = (MutableSet, )

try:
    import numpy
    HAS_NUMPY = True
    LIST_TYPES += (numpy.ndarray, )
except ImportError:  # pragma: no cover
    HAS_NUMPY = False


class DiffResult(object):
    """Iterator over diff operations with an optional ignored report.

    The object is iterable exactly like the previous generator returned by
    :func:`diff`.  When ``report_ignored=True`` is used, the suppressed
    changes are available through :attr:`ignored` after iteration.
    """

    def __init__(self, iterator, ignored=None):
        """Wrap *iterator* and expose the optional *ignored* list."""
        self._iterator = iter(iterator)
        self.ignored = ignored

    @property
    def ignored_changes(self):
        """Return the ignored-change report."""
        return self.ignored

    def __iter__(self):
        """Return the result iterator."""
        return self

    def __next__(self):
        """Return the next diff operation."""
        return next(self._iterator)

    def close(self):
        """Close the underlying iterator when supported."""
        close = getattr(self._iterator, 'close', None)
        if close is not None:
            close()


def _ignored_sinks(report_ignored, ignored_changes):
    """Create report sinks without inspecting any diff input data."""
    internal = None
    sinks = []
    sink_ids = set()

    def add_sink(value):
        if value is None or value is False:
            return
        if value is True:
            return
        if id(value) in sink_ids:
            return
        if hasattr(value, 'append'):
            sinks.append(value.append)
        elif hasattr(value, 'add'):
            sinks.append(value.add)
        elif callable(value):
            sinks.append(value)
        else:
            raise TypeError(
                'ignored change reports must be lists or callables'
            )
        sink_ids.add(id(value))

    if report_ignored is True:
        internal = []
        sinks.append(internal.append)
    else:
        add_sink(report_ignored)
    add_sink(ignored_changes)

    def emit(record):
        for sink in sinks:
            sink(record)

    return internal, emit, bool(sinks)


def diff(first, second, node=None, ignore=None, path_limit=None, expand=False,
         tolerance=EPSILON, absolute_tolerance=None, dot_notation=True,
         numeric_policy=None, report_ignored=None, ignored_changes=None,
         **policy_aliases):
    """Compare two dictionary/list/set objects, and returns a diff result.

    Return an iterator with differences between two objects. The diff items
    represent addition/deletion/change and the item value is a *deep copy*
    from the corresponding source or destination objects.

    >>> from dictdiffer import diff
    >>> result = diff({'a': 'b'}, {'a': 'c'})
    >>> list(result)
    [('change', 'a', ('b', 'c'))]

    The keys can be skipped from difference calculation when they are included
    in ``ignore`` argument of type :class:`collections.Container`.

    >>> list(diff({'a': 1, 'b': 2}, {'a': 3, 'b': 4}, ignore=set(['a'])))
    [('change', 'b', (2, 4))]
    >>> class IgnoreCase(set):
    ...     def __contains__(self, key):
    ...         return set.__contains__(self, str(key).lower())
    >>> list(diff({'a': 1, 'b': 2}, {'A': 3, 'b': 4}, ignore=IgnoreCase('a')))
    [('change', 'b', (2, 4))]

    The difference calculation can be limitted to certain path:

    >>> list(diff({}, {'a': {'b': 'c'}}))
    [('add', '', [('a', {'b': 'c'})])]

    >>> from dictdiffer.utils import PathLimit
    >>> list(diff({}, {'a': {'b': 'c'}}, path_limit=PathLimit()))
    [('add', '', [('a', {})]), ('add', 'a', [('b', 'c')])]

    >>> from dictdiffer.utils import PathLimit
    >>> list(diff({}, {'a': {'b': 'c'}}, path_limit=PathLimit([('a',)])))
    [('add', '', [('a', {'b': 'c'})])]

    >>> from dictdiffer.utils import PathLimit
    >>> list(diff({}, {'a': {'b': 'c'}},
    ...           path_limit=PathLimit([('a', 'b')])))
    [('add', '', [('a', {})]), ('add', 'a', [('b', 'c')])]

    >>> from dictdiffer.utils import PathLimit
    >>> list(diff({'a': {'b': 'c'}}, {'a': {'b': 'c'}}, path_limit=PathLimit([('a',)])))
    []

    The patch can be expanded to small units e.g. when adding multiple values:

    >>> list(diff({'fruits': []}, {'fruits': ['apple', 'mango']}))
    [('add', 'fruits', [(0, 'apple'), (1, 'mango')])]

    >>> list(diff({'fruits': []}, {'fruits': ['apple', 'mango']}, expand=True))
    [('add', 'fruits', [(0, 'apple')]), ('add', 'fruits', [(1, 'mango')])]

    >>> list(diff({'a': {'x': 1}}, {'a': {'x': 2}}))
    [('change', 'a.x', (1, 2))]

    >>> list(diff({'a': {'x': 1}}, {'a': {'x': 2}},
    ... dot_notation=False))
    [('change', ['a', 'x'], (1, 2))]

    :param first: The original dictionary, ``list`` or ``set``.
    :param second: New dictionary, ``list`` or ``set``.
    :param node: Key for comparison that can be used in :func:`dot_lookup`.
    :param ignore: Set of keys that should not be checked.
    :param path_limit: List of path limit tuples or dictdiffer.utils.Pathlimit
                       object to limit the diff recursion depth.
                       A diff is still performed beyond the path_limit,
                       but individual differences will be aggregated up to the path_limit.
    :param expand: Expand the patches.
    :param tolerance: Relative threshold to consider when comparing two
                      numeric values.
    :param absolute_tolerance: Absolute threshold to consider when comparing
                               two numeric values.
    :param dot_notation: Boolean to toggle dot notation on and off.
    :param numeric_policy: A :class:`dictdiffer.numeric.NumericPolicy` or a
                           mapping of structured path patterns to numeric
                           rules.  Matched paths use the policy rule; unmatched
                           paths fall back to the policy default (or the
                           legacy ``tolerance`` values when a raw mapping is
                           supplied).
    :param report_ignored: If ``True``, collect ignored changes on the
                           returned :class:`DiffResult` object.  A list or a
                           callable can also be supplied as a report sink.
    :param ignored_changes: An additional list or callable report sink.

    .. versionchanged:: 0.3
       Added *ignore* parameter.

    .. versionchanged:: 0.4
       Arguments ``first`` and ``second`` can now contain a ``set``.

    .. versionchanged:: 0.5
       Added *path_limit* parameter.
       Added *expand* paramter.
       Added *tolerance* parameter.

    .. versionchanged:: 0.7
       Diff items are deep copies from its corresponding objects.
       Argument *ignore* is always converted to a ``set``.

    .. versionchanged:: 0.8
        Added *dot_notation* parameter.

    .. versionchanged:: 0.11
        Added *numeric_policy*, *report_ignored* and *ignored_changes*
        parameters.
    """
    numeric_policy = take_policy_alias(numeric_policy, policy_aliases)

    if numeric_policy is None and tolerance is not None and \
            not is_numeric(tolerance):
        numeric_policy = tolerance
        tolerance = EPSILON

    policy = coerce_numeric_policy(
        numeric_policy,
        tolerance=tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    internal_ignored, emit_ignored, report_enabled = _ignored_sinks(
        report_ignored, ignored_changes
    )

    if path_limit is not None and not isinstance(path_limit, PathLimit):
        path_limit = PathLimit(path_limit)

    if isinstance(ignore, Iterable):
        def _process_ignore_value(value):
            if isinstance(value, int):
                return value,
            elif isinstance(value, list):
                return tuple(value)
            elif not dot_notation and isinstance(value, str):
                return value,
            return value

        ignore = type(ignore)(_process_ignore_value(value) for value in ignore)

    def dotted(node, default_type=list):
        """Return dotted notation."""
        if dot_notation and \
            all(map(lambda x: isinstance(x, str) and '.' not in x,
                node)):
            return '.'.join(node)
        else:
            return default_type(node)

    def same(path, first_value, second_value):
        """Use the policy for every equality decision."""
        comparison = policy.compare(path, first_value, second_value)
        if report_enabled and comparison.ignored:
            emit_ignored(IgnoredChange(
                path=tuple(path),
                first=deepcopy(first_value),
                second=deepcopy(second_value),
                reason=comparison.reason,
                rule=comparison.rule,
                pattern=comparison.pattern,
                difference=comparison.difference,
                allowed=comparison.allowed,
            ))
        return comparison.equal

    def missing_is_ignored(path, value, missing_side='first'):
        """Apply the rule's missing-field behaviour."""
        matched = policy.resolve(path)
        if matched.rule.missing != 'zero':
            return False
        comparison = policy.compare(path, 0, value)
        if not comparison.equal:
            return False
        if report_enabled:
            if missing_side == 'first':
                first_value, second_value = 0, deepcopy(value)
            else:
                first_value, second_value = deepcopy(value), 0
            emit_ignored(IgnoredChange(
                path=tuple(path),
                first=first_value,
                second=second_value,
                reason='missing-treated-as-zero',
                rule=comparison.rule,
                pattern=comparison.pattern,
                difference=comparison.difference,
                allowed=comparison.allowed,
            ))
        return True

    def same_tree(path, first_value, second_value):
        """Compare an aggregate using the same policy at every leaf."""
        if isinstance(first_value, DICT_TYPES) and \
                isinstance(second_value, DICT_TYPES):
            for key in [k for k in first_value if k not in second_value]:
                if not missing_is_ignored(
                        path + [key], first_value[key], missing_side='second'):
                    return False
            for key in [k for k in second_value if k not in first_value]:
                if not missing_is_ignored(
                        path + [key], second_value[key]):
                    return False
            return all(
                same_tree(path + [key], first_value[key], second_value[key])
                for key in first_value if key in second_value
            )
        if isinstance(first_value, LIST_TYPES) and \
                isinstance(second_value, LIST_TYPES):
            common = min(len(first_value), len(second_value))
            for index in range(common, len(first_value)):
                if not missing_is_ignored(
                        path + [index], first_value[index],
                        missing_side='second'):
                    return False
            for index in range(common, len(second_value)):
                if not missing_is_ignored(
                        path + [index], second_value[index]):
                    return False
            return all(
                same_tree(path + [index], first_value[index],
                          second_value[index])
                for index in range(common)
            )
        if isinstance(first_value, SET_TYPES) and \
                isinstance(second_value, SET_TYPES):
            return first_value == second_value
        return same(path, first_value, second_value)

    def _diff_recursive(_first, _second, _node=None):
        _node = _node or []

        dotted_node = dotted(_node)

        differ = False

        if isinstance(_first, DICT_TYPES) and isinstance(_second, DICT_TYPES):
            # dictionaries are not hashable, we can't use sets
            def check(key):
                """Test if key in current node should be ignored."""
                return ignore is None or (
                    dotted(_node + [key], default_type=tuple) not in ignore and
                    tuple(_node + [key]) not in ignore
                )

            intersection = [k for k in _first if k in _second and check(k)]
            addition = [
                k for k in _second
                if k not in _first and check(k) and
                not missing_is_ignored(_node + [k], _second[k])
            ]
            deletion = [
                k for k in _first
                if k not in _second and check(k) and
                not missing_is_ignored(
                    _node + [k], _first[k], missing_side='second'
                )
            ]

            differ = True

        elif isinstance(_first, LIST_TYPES) and isinstance(_second,
                                                           LIST_TYPES):
            len_first = len(_first)
            len_second = len(_second)

            intersection = list(range(0, min(len_first, len_second)))
            addition = [
                index for index in
                range(min(len_first, len_second), len_second)
                if not missing_is_ignored(_node + [index], _second[index])
            ]
            deletion = [
                index for index in
                reversed(range(min(len_first, len_second), len_first))
                if not missing_is_ignored(
                    _node + [index], _first[index], missing_side='second'
                )
            ]

            differ = True

        elif isinstance(_first, SET_TYPES) and isinstance(_second, SET_TYPES):
            # Deep copy is not necessary for hashable items.
            addition = _second - _first
            if len(addition):
                yield ADD, dotted_node, [(0, addition)]
            deletion = _first - _second
            if len(deletion):
                yield REMOVE, dotted_node, [(0, deletion)]

            return  # stop here for sets

        if differ:
            # Compare if object is a dictionary or list.
            #
            # NOTE variables: intersection, addition, deletion contain only
            # hashable types, hence they do not need to be deepcopied.
            #
            # Call again the parent function as recursive if dictionary have
            # child objects.  Yields `add` and `remove` flags.
            for key in intersection:
                # if type is not changed,
                # callees again diff function to compare.
                # otherwise, the change will be handled as `change` flag.
                if path_limit and path_limit.path_is_limit(_node + [key]):
                    if same_tree(_node + [key],
                                 _first[key], _second[key]):
                        continue

                    yield CHANGE, _node + [key], (
                        deepcopy(_first[key]), deepcopy(_second[key])
                    )
                else:
                    recurred = _diff_recursive(
                        _first[key], _second[key],
                        _node=_node + [key],
                    )

                    for diffed in recurred:
                        yield diffed

            if addition:
                if path_limit:
                    collect = []
                    collect_recurred = []
                    for key in addition:
                        if not isinstance(_second[key],
                                          SET_TYPES + LIST_TYPES + DICT_TYPES):
                            collect.append((key, deepcopy(_second[key])))
                        elif path_limit.path_is_limit(_node + [key]):
                            collect.append((key, deepcopy(_second[key])))
                        else:
                            collect.append((key, _second[key].__class__()))
                            recurred = _diff_recursive(
                                _second[key].__class__(),
                                _second[key],
                                _node=_node + [key],
                            )

                            collect_recurred.append(recurred)

                    if expand:
                        for key, val in collect:
                            yield ADD, dotted_node, [(key, val)]
                    else:
                        yield ADD, dotted_node, collect

                    for recurred in collect_recurred:
                        for diffed in recurred:
                            yield diffed
                else:
                    if expand:
                        for key in addition:
                            yield ADD, dotted_node, [
                                (key, deepcopy(_second[key]))]
                    else:
                        yield ADD, dotted_node, [
                            # for additions, return a list that consist with
                            # two-pair tuples.
                            (key, deepcopy(_second[key])) for key in addition]

            if deletion:
                if expand:
                    for key in deletion:
                        yield REMOVE, dotted_node, [
                            (key, deepcopy(_first[key]))]
                else:
                    yield REMOVE, dotted_node, [
                        # for deletions, return the list of removed keys
                        # and values.
                        (key, deepcopy(_first[key])) for key in deletion]

        else:
            # Compare string and numerical types and yield `change` flag.
            if not same(_node, _first, _second):
                yield CHANGE, dotted_node, (deepcopy(_first),
                                            deepcopy(_second))

    return DiffResult(
        _diff_recursive(first, second, node),
        ignored=internal_ignored,
    )


def patch(diff_result, destination, in_place=False):
    """Patch the diff result to the destination dictionary.

    :param diff_result: Changes returned by ``diff``.
    :param destination: Structure to apply the changes to.
    :param in_place: By default, destination dictionary is deep copied
                     before applying the patch, and the copy is returned.
                     Setting ``in_place=True`` means that patch will apply
                     the changes directly to and return the destination
                     structure.
    """
    if not in_place:
        destination = deepcopy(destination)

    def add(node, changes):
        for key, value in changes:
            dest = dot_lookup(destination, node)
            if isinstance(dest, LIST_TYPES):
                dest.insert(key, value)
            elif isinstance(dest, SET_TYPES):
                dest |= value
            else:
                dest[key] = value

    def change(node, changes):
        dest = dot_lookup(destination, node, parent=True)
        if isinstance(node, str):
            last_node = node.split('.')[-1]
        else:
            last_node = node[-1]
        if isinstance(dest, LIST_TYPES):
            last_node = int(last_node)
        _, value = changes
        dest[last_node] = value

    def remove(node, changes):
        for key, value in changes:
            dest = dot_lookup(destination, node)
            if isinstance(dest, SET_TYPES):
                dest -= value
            else:
                del dest[key]

    patchers = {
        REMOVE: remove,
        ADD: add,
        CHANGE: change
    }

    for action, node, changes in diff_result:
        patchers[action](node, changes)

    return destination


def swap(diff_result):
    """Swap the diff result.

    It uses following mapping:

    - remove -> add
    - add -> remove

    In addition, swap the changed values for `change` flag.

        >>> from dictdiffer import swap
        >>> swapped = swap([('add', 'a.b.c', [('a', 'b'), ('c', 'd')])])
        >>> next(swapped)
        ('remove', 'a.b.c', [('c', 'd'), ('a', 'b')])

        >>> swapped = swap([('change', 'a.b.c', ('a', 'b'))])
        >>> next(swapped)
        ('change', 'a.b.c', ('b', 'a'))

    """
    def add(node, changes):
        return REMOVE, node, list(reversed(changes))

    def remove(node, changes):
        return ADD, node, changes

    def change(node, changes):
        first, second = changes
        return CHANGE, node, (second, first)

    swappers = {
        REMOVE: remove,
        ADD: add,
        CHANGE: change
    }

    for action, node, change in diff_result:
        yield swappers[action](node, change)


def revert(diff_result, destination, in_place=False):
    """Call swap function to revert patched dictionary object.

    Usage example:

        >>> from dictdiffer import diff, revert
        >>> first = {'a': 'b'}
        >>> second = {'a': 'c'}
        >>> revert(diff(first, second), second)
        {'a': 'b'}

    :param diff_result: Changes returned by ``diff``.
    :param destination: Structure to apply the changes to.
    :param in_place: By default, destination dictionary is deep
                     copied before being reverted, and the copy
                     is returned. Setting ``in_place=True`` means
                     that revert will apply the changes directly to
                     and return the destination structure.
    """
    return patch(swap(diff_result), destination, in_place)
