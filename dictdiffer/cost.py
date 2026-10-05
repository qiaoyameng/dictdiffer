"""Deterministic cost helpers for selecting diff operation granularity."""

from collections.abc import MutableMapping, MutableSequence, MutableSet
from copy import deepcopy

(ADD, REMOVE, CHANGE) = ('add', 'remove', 'change')
DICT_TYPES = (MutableMapping, )
SET_TYPES = (MutableSet, )

try:
    import numpy
    LIST_TYPES = (MutableSequence, numpy.ndarray)
except ImportError:  # pragma: no cover
    LIST_TYPES = (MutableSequence, )

CONTAINER_TYPES = DICT_TYPES + LIST_TYPES + SET_TYPES
COLLECTION_TYPES = DICT_TYPES + LIST_TYPES


def _canonical(value):
    """Return a deterministic representation for cost encoding."""
    if isinstance(value, DICT_TYPES):
        return dict(sorted((
            (_canonical(key), _canonical(item))
            for key, item in value.items()
        ), key=repr))
    if isinstance(value, MutableSet):
        return ['set', *sorted(map(_canonical, value), key=repr)]
    if isinstance(value, frozenset):
        return ['frozenset', *sorted(map(_canonical, value), key=repr)]
    if isinstance(value, tuple):
        return tuple(map(_canonical, value))
    if isinstance(value, LIST_TYPES):
        return [_canonical(item) for item in value]
    return value


def byte_cost(operation):
    """Return the UTF-8 byte length of a deterministic operation encoding."""
    return len(repr(_canonical(operation)).encode('utf-8'))


class _Plan(object):
    """Operations selected for a subtree and limits affecting its ancestors."""

    def __init__(self, operations=None, blocked=False, ignored=False):
        self.operations = operations or []
        self.blocked = blocked
        self.ignored = ignored


def cost_diff(first, second, dotted, is_ignored, path_limit,
              different, expand, cost_func, node=None):
    """Compare structures while choosing mutually exclusive granularities."""
    def contains_ignored(value, path):
        if isinstance(value, DICT_TYPES):
            return any(
                is_ignored(path + [key]) or
                contains_ignored(value[key], path + [key])
                for key in value
            )
        if isinstance(value, LIST_TYPES):
            return any(
                contains_ignored(item, path + [index])
                for index, item in enumerate(value)
            )
        return False

    def contains_path_limit(value, path):
        if not path_limit:
            return False
        if isinstance(value, DICT_TYPES):
            children = list(value)
        elif isinstance(value, LIST_TYPES):
            children = range(len(value))
        else:
            return False
        return any(
            path_limit.path_is_limit(path + [key]) or
            contains_path_limit(value[key], path + [key])
            for key in children
        )

    def included_value(value, path):
        """Copy an added value without any path selected for ignoring."""
        if isinstance(value, DICT_TYPES):
            result = value.__class__()
            for key in value:
                child_path = path + [key]
                if not is_ignored(child_path):
                    included = included_value(value[key], child_path)
                    if not (isinstance(included, DICT_TYPES) and
                            not included):
                        result[key] = included
            return result
        if isinstance(value, LIST_TYPES):
            items = [
                included_value(item, path + [index])
                for index, item in enumerate(value)
            ]
            if isinstance(value, MutableSequence):
                result = value.__class__()
                result.extend(items)
            else:
                result = value.__class__(items)
            return result
        return deepcopy(value)

    def recursive(first_value, second_value, path=None, exists=True):
        path = path or []
        current_node = dotted(path)
        operations = []
        blocked = False
        ignored = False

        def append_changes(action, changes):
            if not changes:
                return
            if expand:
                for key, value in changes:
                    operations.append(
                        (action, current_node, [(key, deepcopy(value))])
                    )
            else:
                operations.append(
                    (action, current_node,
                     [(key, deepcopy(value)) for key, value in changes])
                )

        if isinstance(first_value, SET_TYPES) and isinstance(
                second_value, SET_TYPES):
            addition = second_value - first_value
            deletion = first_value - second_value
            if addition:
                operations.append(
                    (ADD, current_node, [(0, deepcopy(addition))])
                )
            if deletion:
                operations.append(
                    (REMOVE, current_node, [(0, deepcopy(deletion))])
                )
        elif (isinstance(first_value, DICT_TYPES) and
              isinstance(second_value, DICT_TYPES)) or (
                isinstance(first_value, LIST_TYPES) and
                isinstance(second_value, LIST_TYPES)):
            if isinstance(first_value, DICT_TYPES):
                def check(key):
                    return not is_ignored(path + [key])

                intersection = [key for key in first_value
                                if key in second_value and check(key)]
                addition = [key for key in second_value
                            if key not in first_value and check(key)]
                deletion = [key for key in first_value
                            if key not in second_value and check(key)]
                if any(not check(key)
                       for key in list(first_value) + list(second_value)):
                    blocked = True
                    ignored = True
            else:
                common_length = min(len(first_value), len(second_value))
                intersection = list(range(common_length))
                addition = list(range(common_length, len(second_value)))
                deletion = list(reversed(range(common_length,
                                               len(first_value))))

            for key in intersection:
                child_path = path + [key]
                if path_limit and path_limit.path_is_limit(child_path):
                    blocked = True
                    if (contains_ignored(first_value[key], child_path) or
                            contains_ignored(
                                second_value[key], child_path
                            )):
                        child = recursive(
                            first_value[key], second_value[key], child_path
                        )
                        blocked = blocked or child.blocked
                        ignored = ignored or child.ignored
                        operations.extend(child.operations)
                    elif different(first_value[key], second_value[key]):
                        operations.append(
                            (CHANGE, child_path,
                             (deepcopy(first_value[key]),
                              deepcopy(second_value[key])))
                        )
                    continue

                child = recursive(first_value[key], second_value[key],
                                  child_path)
                blocked = blocked or child.blocked
                ignored = ignored or child.ignored
                operations.extend(child.operations)

            add_changes = []
            add_operations = []
            for key in addition:
                value = second_value[key]
                child_path = path + [key]
                is_container = isinstance(value, CONTAINER_TYPES)
                has_ignored = is_container and contains_ignored(
                    value, child_path
                )
                is_limit = path_limit and path_limit.path_is_limit(
                    child_path
                )

                if is_limit or has_ignored:
                    included = (
                        included_value(value, child_path)
                        if has_ignored else deepcopy(value)
                    )
                    if not (isinstance(included, DICT_TYPES) and
                            not included):
                        add_changes.append((key, included))
                    blocked = True
                    ignored = ignored or has_ignored
                elif is_container and path_limit:
                    child = recursive(value.__class__(), value, child_path,
                                      exists=False)
                    blocked = True
                    ignored = ignored or child.ignored
                    if child.operations:
                        add_changes.append((key, value.__class__()))
                        add_operations.extend(child.operations)
                else:
                    add_changes.append((key, deepcopy(value)))

            append_changes(ADD, add_changes)
            operations.extend(add_operations)

            remove_changes = []
            remove_operations = []
            for key in deletion:
                value = first_value[key]
                child_path = path + [key]
                is_container = isinstance(value, CONTAINER_TYPES)
                has_ignored = is_container and contains_ignored(
                    value, child_path
                )
                if has_ignored:
                    child = recursive(value, value.__class__(), child_path)
                    blocked = True
                    ignored = True
                    remove_operations.extend(child.operations)
                else:
                    remove_changes.append((key, deepcopy(value)))
                    if path_limit.path_is_limit(child_path) or (
                            is_container and
                            contains_path_limit(value, child_path)):
                        blocked = True

            remove_operations.extend(operations)
            operations = remove_operations
            append_changes(REMOVE, remove_changes)
        elif different(first_value, second_value):
            operations.append(
                (CHANGE, current_node,
                 (deepcopy(first_value), deepcopy(second_value)))
            )

        if operations and exists and not blocked:
            replacement = (
                CHANGE, current_node,
                (deepcopy(first_value), deepcopy(second_value))
            )
            if cost_func(replacement) < sum(
                    cost_func(operation) for operation in operations):
                operations = [replacement]

        return _Plan(operations, blocked, ignored)

    return iter(recursive(first, second, node).operations)
