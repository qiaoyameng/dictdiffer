"""Cost-aware selection of diff granularity.

When many fields of a nested object change together, emitting a
fine-grained diff can be larger than replacing the whole subtree.  This
module implements an explicit cost model: the caller provides a
deterministic operation encoding cost function and
:func:`diff_optimal` picks, for every changed subtree, between the usual
fine-grained ``add``/``remove``/``change`` operations and a single
``change`` operation replacing the complete subtree.
"""

from collections.abc import Iterable, MutableMapping, MutableSequence, MutableSet
from copy import deepcopy

from .utils import EPSILON, PathLimit, are_different

(ADD, REMOVE, CHANGE) = ('add', 'remove', 'change')

DICT_TYPES = (MutableMapping, )
LIST_TYPES = (MutableSequence, )
SET_TYPES = (MutableSet, )

try:
    import numpy
    LIST_TYPES += (numpy.ndarray, )
except ImportError:  # pragma: no cover
    pass


def byte_cost(action, node, changes):
    """Return a simple deterministic byte cost for a patch operation.

    The operation is encoded as ``'<action> <path> <payload>'`` where the
    payload uses :func:`repr` and the cost is the UTF-8 byte length of
    the result.  It is a self-contained example of a cost function and
    is not tied to any serialization or transport library.

    Note that the determinism of the default relies on ``repr`` of the
    involved values; for values with unstable representations (e.g.
    ``set`` under hash randomization) provide a custom ``cost``
    function instead.

    >>> byte_cost('change', 'a.x', (1, 2))
    17

    :param action: The operation, one of ``add``, ``remove``, ``change``.
    :param node: The patch node, a dotted string or a list of keys.
    :param changes: The operation payload as it appears in the patch.
    """
    if isinstance(node, str):
        path = node
    else:
        path = '.'.join(map(str, node))
    return len('{0} {1} {2!r}'.format(action, path, changes)
               .encode('utf-8'))


def diff_optimal(first, second, cost=None, node=None, ignore=None,
                 path_limit=None, tolerance=EPSILON,
                 absolute_tolerance=None, dot_notation=True):
    """Compare two objects and return a minimum-cost patch.

    Works like :func:`dictdiffer.diff` but chooses, for every changed
    subtree, between the usual fine-grained operations and a single
    ``change`` operation replacing the complete subtree, whichever is
    cheaper according to ``cost``.  A returned plan never replaces a
    node and modifies its children at the same time, and the root node
    itself is never replaced so that the result stays applicable with
    :func:`dictdiffer.patch` and :func:`dictdiffer.revert`.

    ``ignore`` and ``path_limit`` remain hard constraints: a subtree
    that contains ignored keys or a ``path_limit`` boundary is never
    replaced wholesale, so excluded content is never pulled into the
    patch just to save bytes.

    The result is a list (not a generator) of the usual diff tuples and
    can be used with :func:`dictdiffer.patch`, :func:`dictdiffer.swap`
    and :func:`dictdiffer.revert` exactly like the result of
    :func:`dictdiffer.diff`.

    >>> from dictdiffer import diff_optimal, patch, revert
    >>> first = {'a': [1, 2, 3]}
    >>> second = {'a': [4, 5, 6]}
    >>> result = diff_optimal(first, second)
    >>> result
    [('change', 'a', ([1, 2, 3], [4, 5, 6]))]
    >>> patch(result, first) == second
    True
    >>> revert(result, second) == first
    True

    A single changed leaf stays a fine-grained change:

    >>> diff_optimal({'a': {'x': 1, 'y': 2}}, {'a': {'x': 1, 'y': 3}})
    [('change', 'a.y', (2, 3))]

    :param first: The original dictionary, ``list`` or ``set``.
    :param second: New dictionary, ``list`` or ``set``.
    :param cost: Deterministic callable ``cost(action, node, changes)``
                 returning a non-negative number, where the arguments
                 mirror a single patch tuple.  Defaults to
                 :func:`byte_cost`.
    :param node: Key for comparison that can be used in
                 :func:`dictdiffer.dot_lookup`.
    :param ignore: Set of keys that should not be checked.
    :param path_limit: List of path limit tuples or
                       dictdiffer.utils.PathLimit object to limit the
                       diff recursion depth.
    :param tolerance: Threshold to consider when comparing two float
                      numbers.
    :param absolute_tolerance: Absolute threshold to consider when
                               comparing two float numbers.
    :param dot_notation: Boolean to toggle dot notation on and off.

    .. versionadded:: 0.10.1
    """
    if cost is None:
        cost = byte_cost

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

        ignore = type(ignore)(_process_ignore_value(value)
                              for value in ignore)

    def dotted(_node, default_type=list):
        """Return dotted notation."""
        if dot_notation and \
            all(map(lambda x: isinstance(x, str) and '.' not in x,
                _node)):
            return '.'.join(_node)
        else:
            return default_type(_node)

    def check(key, _node):
        """Test if key in current node should be ignored."""
        return ignore is None or (
            dotted(_node + [key], default_type=tuple) not in ignore and
            tuple(_node + [key]) not in ignore
        )

    # Paths that a wholesale subtree replacement must never span, so
    # that excluded content cannot leak into the patch.
    ignore_paths = None
    if ignore is not None:
        ignore_paths = set()
        if isinstance(ignore, Iterable):
            for value in ignore:
                if isinstance(value, tuple):
                    ignore_paths.add(value)
                elif isinstance(value, str):
                    ignore_paths.add(tuple(value.split('.')))
                else:
                    ignore_paths.add((value, ))
        # A non-iterable ignore container cannot be enumerated, hence
        # no subtree replacement is allowed at all (ignore_paths stays
        # None and is treated as "block everywhere").

    limit_paths = set()
    if path_limit is not None:
        def _collect_limits(mapping, prefix):
            for key, value in mapping.items():
                if key == path_limit.final_key:
                    limit_paths.add(prefix)
                else:
                    _collect_limits(value, prefix + (key, ))
        _collect_limits(path_limit.dict, ())

    def replace_allowed(_node):
        """Test if the subtree at _node may be replaced wholesale."""
        if ignore is not None and ignore_paths is None:
            return False
        node_tuple = tuple(_node)
        for path in ignore_paths or ():
            # An ignored path at or below _node forbids replacement.
            if len(path) >= len(node_tuple) and \
                    path[:len(node_tuple)] == node_tuple:
                return False
        for path in limit_paths:
            # A path_limit boundary strictly below _node forbids
            # replacement ('*' matches any single key).
            if len(path) > len(node_tuple) and all(
                    limit == '*' or limit == part
                    for limit, part in zip(path, node_tuple)):
                return False
        return True

    def _optimal(_first, _second, _node, is_root=False,
                 allow_replace=True):
        """Return (cost, ops) of the cheapest legal plan at _node."""
        if isinstance(_first, SET_TYPES) and isinstance(_second,
                                                        SET_TYPES):
            ops = []
            total = 0
            addition = _second - _first
            if len(addition):
                payload = [(0, addition)]
                total += cost(ADD, dotted(_node), payload)
                ops.append((ADD, dotted(_node), payload))
            deletion = _first - _second
            if len(deletion):
                payload = [(0, deletion)]
                total += cost(REMOVE, dotted(_node), payload)
                ops.append((REMOVE, dotted(_node), payload))
        elif isinstance(_first, DICT_TYPES) and isinstance(
                _second, DICT_TYPES) or isinstance(
                _first, LIST_TYPES) and isinstance(_second, LIST_TYPES):
            if isinstance(_first, DICT_TYPES):
                intersection = [k for k in _first
                                if k in _second and check(k, _node)]
                addition = [k for k in _second
                            if k not in _first and check(k, _node)]
                deletion = [k for k in _first
                            if k not in _second and check(k, _node)]
            else:
                len_first = len(_first)
                len_second = len(_second)
                intersection = list(range(0, min(len_first, len_second)))
                addition = list(range(min(len_first, len_second),
                                      len_second))
                deletion = list(reversed(range(min(len_first, len_second),
                                               len_first)))

            ops = []
            total = 0

            for key in intersection:
                child = _node + [key]
                if path_limit is not None and \
                        path_limit.path_is_limit(child):
                    if _first[key] != _second[key]:
                        payload = (deepcopy(_first[key]),
                                   deepcopy(_second[key]))
                        total += cost(CHANGE, dotted(child), payload)
                        ops.append((CHANGE, dotted(child), payload))
                else:
                    child_cost, child_ops = _optimal(
                        _first[key], _second[key], child)
                    total += child_cost
                    ops.extend(child_ops)

            if addition:
                if path_limit is not None:
                    collect = []
                    recurred = []
                    for key in addition:
                        if not isinstance(
                                _second[key],
                                SET_TYPES + LIST_TYPES + DICT_TYPES):
                            collect.append((key, deepcopy(_second[key])))
                        elif path_limit.path_is_limit(_node + [key]):
                            collect.append((key, deepcopy(_second[key])))
                        else:
                            collect.append((key,
                                            _second[key].__class__()))
                            recurred.append(_optimal(
                                _second[key].__class__(), _second[key],
                                _node + [key], allow_replace=False))
                    if collect:
                        total += cost(ADD, dotted(_node), collect)
                        ops.append((ADD, dotted(_node), collect))
                    for child_cost, child_ops in recurred:
                        total += child_cost
                        ops.extend(child_ops)
                else:
                    payload = [(key, deepcopy(_second[key]))
                               for key in addition]
                    total += cost(ADD, dotted(_node), payload)
                    ops.append((ADD, dotted(_node), payload))

            if deletion:
                payload = [(key, deepcopy(_first[key]))
                           for key in deletion]
                total += cost(REMOVE, dotted(_node), payload)
                ops.append((REMOVE, dotted(_node), payload))
        else:
            # Compare string and numerical types, yield `change` flag.
            if are_different(_first, _second, tolerance,
                             absolute_tolerance):
                payload = (deepcopy(_first), deepcopy(_second))
                return cost(CHANGE, dotted(_node), payload), [
                    (CHANGE, dotted(_node), payload)]
            return 0, []

        if allow_replace and not is_root and ops and \
                replace_allowed(_node):
            payload = (deepcopy(_first), deepcopy(_second))
            replace_cost = cost(CHANGE, dotted(_node), payload)
            if replace_cost < total:
                return replace_cost, [(CHANGE, dotted(_node), payload)]
        return total, ops

    _, ops = _optimal(first, second, node or [], is_root=True)
    return ops
