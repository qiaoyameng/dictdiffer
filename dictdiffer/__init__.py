# SPDX-FileCopyrightText: 2013 Fatih Erikli.
# SPDX-FileCopyrightText: 2013, 2014, 2015, 2016 CERN.
# SPDX-FileCopyrightText: 2017-2019 ETH Zurich, Swiss Data Science Center, Jiri Kuncar.
# SPDX-License-Identifier: MIT

"""Dictdiffer is a helper module to diff and patch dictionaries."""

from collections.abc import Iterable, MutableMapping, MutableSequence, MutableSet
from copy import deepcopy

from .conflict import PatchConflict
from .utils import EPSILON, PathLimit, are_different, dot_lookup
from .version import __version__

(ADD, REMOVE, CHANGE) = (
    'add', 'remove', 'change')

__all__ = ('diff', 'patch', 'checked_patch', 'swap', 'revert', 'dot_lookup',
           'PatchConflict', 'PatchApplyError', '__version__')

DICT_TYPES = (MutableMapping, )
LIST_TYPES = (MutableSequence, )
SET_TYPES = (MutableSet, )

try:
    import numpy
    HAS_NUMPY = True
    LIST_TYPES += (numpy.ndarray, )
except ImportError:  # pragma: no cover
    HAS_NUMPY = False


class _Missing(object):
    """Sentinel marking a value that is absent from the destination.

    The ``None`` value is a perfectly valid dictionary value, therefore it
    cannot be used to represent a missing path. Conflict reasons use this
    sentinel instead, so that a missing value is never displayed as
    ``None``.
    """

    def __repr__(self):
        return '<missing>'


MISSING = _Missing()


class PatchApplyError(RuntimeError):
    """Raised when a conflict-free patch cannot be applied.

    All structural conflicts are reported *before* the destination is
    modified, therefore this error can only be caused by a custom container
    that cannot be deep copied for the check, or whose own mutating method
    (``__setitem__``, ``insert``, ``__delitem__`` ...) raises when the
    validated patch is applied in place. The structural check cannot undo
    arbitrary external side effects performed by such a custom container.
    """


def _node_keys(node):
    """Return the literal path components of a patch node.

    String nodes keep the historical dot notation and are split on ``.``;
    list/tuple nodes are taken literally, therefore a key containing dots is
    never accidentally split.
    """
    if node is None or node == '' or node == []:
        return []
    if isinstance(node, str):
        return node.split('.')
    return list(node)


def _canonical_path(node, key=MISSING):
    """Build the canonical path tuple of a node and an optional last key."""
    path = _node_keys(node)
    if key is not MISSING:
        path.append(key)
    return tuple(path)


def _traverse(destination, keys):
    """Walk ``keys`` inside ``destination``.

    Returns the parent container, the last key (already converted to an
    integer for sequences) and the current value, or raises :class:`KeyError`
    when a path component does not exist and :class:`TypeError` when a
    container type does not support the traversal.
    """
    parent = destination
    value = destination
    last_key = MISSING
    for component in keys:
        parent = value
        try:
            key = int(component) if isinstance(value, LIST_TYPES) \
                else component
            last_key = key
            value = value[key]
        except (TypeError, ValueError):
            # A non integer index used on a sequence means that the
            # container type changed underneath the patch node.
            raise TypeError(
                'cannot resolve path component {0!r} inside {1} ({2}): '
                'the container type changed'.format(
                    component, value, type(value).__name__))
    return parent, last_key, value


def _mismatch_reason(expected, actual):
    """Build the reason text for an expected/actual value mismatch."""
    if type(expected) is not type(actual):
        return ('type changed: expected old value {0!r} ({1}), found '
                '{2!r} ({3})').format(expected, type(expected).__name__,
                                      actual, type(actual).__name__)
    return 'expected old value {0!r}, found {1!r}'.format(
        expected, actual)


def _checked_change(index, destination, node, changes):
    """Check and apply a ``change`` operation against ``destination``."""
    old_value, new_value = changes
    keys = _node_keys(node)
    path = _canonical_path(node)
    try:
        parent, last_key, current = _traverse(destination, keys)
    except KeyError:
        return [PatchConflict(
            index, path,
            'path does not exist; expected old value {0!r}'.format(
                old_value))]
    except (IndexError, TypeError, ValueError) as error:
        return [PatchConflict(index, path, str(error))]
    if current != old_value:
        return [PatchConflict(
            index, path, _mismatch_reason(old_value, current))]
    try:
        parent[last_key] = new_value
    except Exception as error:  # custom mapping may reject the assignment
        return [PatchConflict(index, path,
                              'cannot assign value: {0}'.format(error))]
    return []


def _checked_add(index, destination, node, changes):
    """Check and apply an ``add`` operation against ``destination``."""
    conflicts = []
    try:
        container = dot_lookup(destination, node)
    except (KeyError, IndexError, TypeError, ValueError) as error:
        for key, _value in changes:
            conflicts.append(PatchConflict(
                index, _canonical_path(node, key), str(error)))
        return conflicts
    for key, value in changes:
        path = _canonical_path(node, key)
        if isinstance(container, SET_TYPES):
            already_present = [item for item in value if item in container]
            if already_present:
                conflicts.append(PatchConflict(
                    index, path,
                    'add target already contains {0!r}'.format(
                        already_present)))
                continue
            try:
                container |= value
            except Exception as error:
                conflicts.append(PatchConflict(
                    index, path,
                    'cannot add value: {0}'.format(error)))
        elif isinstance(container, LIST_TYPES):
            if not isinstance(key, int) or key < 0 or key > len(container):
                conflicts.append(PatchConflict(
                    index, path,
                    'list insert index {0!r} is out of range (length {1})'
                    .format(key, len(container))))
                continue
            try:
                container.insert(key, value)
            except Exception as error:
                conflicts.append(PatchConflict(
                    index, path,
                    'cannot insert value: {0}'.format(error)))
        elif isinstance(container, DICT_TYPES):
            if key in container:
                conflicts.append(PatchConflict(
                    index, path,
                    'add target already exists with value {0!r}'.format(
                        container[key])))
                continue
            try:
                container[key] = value
            except Exception as error:
                conflicts.append(PatchConflict(
                    index, path,
                    'cannot add value: {0}'.format(error)))
        else:
            conflicts.append(PatchConflict(
                index, path,
                'target is not a container ({0})'.format(
                    type(container).__name__)))
    return conflicts


def _checked_remove(index, destination, node, changes):
    """Check and apply a ``remove`` operation against ``destination``."""
    conflicts = []
    try:
        container = dot_lookup(destination, node)
    except (KeyError, IndexError, TypeError, ValueError) as error:
        for key, _value in changes:
            conflicts.append(PatchConflict(
                index, _canonical_path(node, key), str(error)))
        return conflicts
    for key, value in changes:
        path = _canonical_path(node, key)
        if isinstance(container, SET_TYPES):
            missing = [item for item in value if item not in container]
            if missing:
                conflicts.append(PatchConflict(
                    index, path,
                    'remove target {0!r} is missing'.format(missing)))
                continue
            try:
                container -= value
            except Exception as error:
                conflicts.append(PatchConflict(
                    index, path,
                    'cannot remove value: {0}'.format(error)))
        elif isinstance(container, LIST_TYPES):
            if not isinstance(key, int) or key < 0 or key >= \
                    len(container):
                conflicts.append(PatchConflict(
                    index, path,
                    'list index {0!r} is out of range (length {1})'.format(
                        key, len(container))))
                continue
            if container[key] != value:
                conflicts.append(PatchConflict(
                    index, path,
                    _mismatch_reason(value, container[key])))
                continue
            try:
                del container[key]
            except Exception as error:
                conflicts.append(PatchConflict(
                    index, path,
                    'cannot remove value: {0}'.format(error)))
        elif isinstance(container, DICT_TYPES):
            if key not in container:
                conflicts.append(PatchConflict(
                    index, path,
                    'remove target is missing; expected old value {0!r}'
                    .format(value)))
                continue
            if container[key] != value:
                conflicts.append(PatchConflict(
                    index, path,
                    _mismatch_reason(value, container[key])))
                continue
            try:
                del container[key]
            except Exception as error:
                conflicts.append(PatchConflict(
                    index, path,
                    'cannot remove value: {0}'.format(error)))
        else:
            conflicts.append(PatchConflict(
                index, path,
                'target is not a container ({0})'.format(
                    type(container).__name__)))
    return conflicts


_CHECKED_PATCHERS = {
    ADD: _checked_add,
    REMOVE: _checked_remove,
    CHANGE: _checked_change,
}


def checked_patch(diff_result, destination, in_place=False):
    """Check a patch against the destination and apply it when it is current.

    Every operation is verified on a working copy before the destination is
    touched:

    * ``change`` and ``remove`` operations must find their recorded old
      value at the expected location.
    * ``add`` operations must target an existing container that does not yet
      hold the added key/index/item and accept the new value.

    Operations are interpreted sequentially, therefore list indices refer to
    the state *at the moment the operation runs* and earlier operations are
    already visible to later ones. Nodes given as lists or tuples are treated
    literally, so dictionary keys containing dots are never split by the dot
    notation.

    All locatable conflicts are collected in one pass. When there is at least
    one conflict nothing is applied and the destination stays unchanged, both
    for the default copy mode and for ``in_place=True``. Each
    :class:`PatchConflict` carries the zero based operation number, the
    canonical path tuple and the concrete reason; a missing value is reported
    as ``<missing>``, never as ``None`` (``None`` may be a real value).

    :param diff_result: Changes returned by :func:`diff`.
    :param destination: Structure to apply the changes to.
    :param in_place: By default the destination is deep copied and the copy
                     is returned; when ``True`` the destination itself is
                     modified and returned.
    :returns: ``(patched_destination, conflicts)`` tuple. When ``conflicts``
              is non empty, ``patched_destination`` is ``None`` and the
              destination is guaranteed to be unchanged.
    :raises PatchApplyError: only when a *custom* container prevents the
                             already validated patch from being processed,
                             either because the destination cannot be deep
                             copied for the check or because its own mutating
                             method rejects the write during an in place
                             application. The structural check cannot roll
                             back arbitrary external side effects performed
                             by such a custom container, therefore no
                             rollback of external side effects is promised
                             beyond leaving the built in containers
                             untouched.
    """
    patches = list(diff_result)
    try:
        working_copy = deepcopy(destination)
    except Exception as error:
        raise PatchApplyError(
            'the destination could not be deep copied for the structural '
            'check: {0}'.format(error))

    conflicts = []
    for index, (action, node, changes) in enumerate(patches):
        try:
            checker = _CHECKED_PATCHERS[action]
        except KeyError:
            raise ValueError('unknown patch action {0!r}'.format(action))
        found = checker(index, working_copy, node, changes)
        conflicts.extend(found)

    if conflicts:
        return None, conflicts

    if not in_place:
        return working_copy, []

    try:
        return patch(patches, destination, in_place=True), []
    except Exception as error:
        raise PatchApplyError(
            'the patch passed all structural checks but could not be '
            'applied in place: {0}'.format(error))


def diff(first, second, node=None, ignore=None, path_limit=None, expand=False,
         tolerance=EPSILON, absolute_tolerance=None, dot_notation=True):
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
    :param tolerance: Threshold to consider when comparing two float numbers.
    :param absolute_tolerance: Absolute threshold to consider when comparing
                               two float numbers.
    :param dot_notation: Boolean to toggle dot notation on and off.

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
    """
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
            addition = [k for k in _second if k not in _first and check(k)]
            deletion = [k for k in _first if k not in _second and check(k)]

            differ = True

        elif isinstance(_first, LIST_TYPES) and isinstance(_second,
                                                           LIST_TYPES):
            len_first = len(_first)
            len_second = len(_second)

            intersection = list(range(0, min(len_first, len_second)))
            addition = list(range(min(len_first, len_second), len_second))
            deletion = list(
                reversed(range(min(len_first, len_second), len_first)))

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
                    if _first[key] == _second[key]:
                        return

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
            if are_different(_first, _second, tolerance, absolute_tolerance):
                yield CHANGE, dotted_node, (deepcopy(_first),
                                            deepcopy(_second))

    return _diff_recursive(first, second, node)


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
