# SPDX-FileCopyrightText: 2026 TU Wien.
# SPDX-License-Identifier: MIT

"""Symbolic composition of consecutive dictdiffer patches.

The public function is :func:`compose`.  It accepts the tuple format produced
by :func:`dictdiffer.diff`:

- ``('change', node, (old_value, new_value))``;
- ``('add', node, [(key, value), ...])``;
- ``('remove', node, [(key, value), ...])``;
- set operations use a set value with the conventional key ``0``.

Both grouped and ``expand=True`` list operations are supported, provided the
operations are in the order in which ``patch`` can apply them.  The result is
also a normal dictdiffer patch and can be passed to ``patch`` and ``revert``.

The composer does not need A, B or C.  It tracks dictionary keys, list item
identities, set membership and the old values contained in the patches.
Contradictory old values, missing paths or invalid list/set preconditions raise
:class:`PatchCompositionError` before any caller document is patched.

Root replacement (``('change', '', ...)``) cannot be represented by the usual
container operations and is not composed.  The set encoding ``(0, value)`` is
also intrinsically ambiguous when the parent is an unknown dictionary or list
with integer key ``0``; such a parent must be established by an earlier
operation.
"""

from collections.abc import MutableMapping, MutableSequence, MutableSet
from copy import deepcopy

from .utils import are_different

ADD = 'add'
REMOVE = 'remove'
CHANGE = 'change'

UNKNOWN = object()
ABSENT = object()


class PatchCompositionError(ValueError):
    """Raised when two consecutive patches cannot be composed."""

    def __init__(self, message, first_patch=None, second_patch=None):
        super(PatchCompositionError, self).__init__(message)
        self.first_patch = first_patch
        self.second_patch = second_patch


class Cell(object):
    """A value tracked at a dictionary key or list position."""

    def __init__(self, initial=UNKNOWN, current=UNKNOWN):
        self.initial = initial
        self.current = current


class SparseDict(object):
    """Known keys of a dictionary in A and current intermediate state."""

    def __init__(self):
        self.initial = {}
        self.current = {}


class SparseList(object):
    """List item identities, without tombstones in the current sequence."""

    def __init__(self):
        self.initial = []
        self.current = []


class SparseSet(object):
    """Known membership information for a set."""

    def __init__(self):
        self.initial = {}
        self.current = {}


def _mapping(value):
    return value is not ABSENT and isinstance(value, MutableMapping)


def _sequence(value):
    return value is not ABSENT and isinstance(value, MutableSequence)


def _set(value):
    return value is not ABSENT and isinstance(value, MutableSet)


def _container(value):
    return _mapping(value) or _sequence(value) or _set(value)


def _same(left, right):
    return not are_different(left, right, 0)


def _different(left, right):
    return are_different(left, right, 0)


def _error(message, op=None, stage=1):
    if op is None:
        return PatchCompositionError(message)
    if stage == 1:
        return PatchCompositionError(message, first_patch=op['original'])
    return PatchCompositionError(message, second_patch=op['original'])


def _path(node):
    if node is None or node == '':
        return []
    if isinstance(node, str):
        return node.split('.')
    if isinstance(node, (list, tuple)):
        return list(node)
    raise PatchCompositionError('Patch node must be a string, list or tuple.')


def _normalize(patch, stage):
    operations = []
    for item in patch:
        try:
            action, node, payload = item
        except (TypeError, ValueError):
            raise PatchCompositionError(
                'Every patch item must be a three-item tuple.')

        path = _path(node)
        original = tuple(item)

        if action == CHANGE:
            try:
                old, new = payload
            except (TypeError, ValueError):
                raise _error('A change payload must contain two values.',
                             {'original': original}, stage)
            operations.append({
                'action': CHANGE,
                'path': path,
                'parent_path': path[:-1],
                'key': None if not path else path[-1],
                'old': deepcopy(old),
                'new': deepcopy(new),
                'tentative_set': False,
                'list_set_value': False,
                'stage': stage,
                'original': original,
            })
            continue

        if action not in (ADD, REMOVE):
            raise PatchCompositionError(
                "Unsupported patch action '{0}'.".format(action))

        try:
            pairs = list(payload)
        except TypeError:
            raise _error('Add and remove payloads must be iterable pairs.',
                         {'original': original}, stage)

        for key, value in pairs:
            tentative_set = key == 0 and isinstance(value, MutableSet)
            list_set_value = tentative_set and path and isinstance(
                path[-1], int)
            if tentative_set and path and isinstance(path[-1], int):
                tentative_set = False
                item_key = path[-1]
                item_parent_path = path[:-1]
            else:
                item_key = key
                item_parent_path = path
            operations.append({
                'action': action,
                'path': path if tentative_set else
                item_parent_path + [item_key],
                'parent_path': path if tentative_set else
                item_parent_path,
                'key': key if tentative_set else item_key,
                'value': deepcopy(value),
                'tentative_set': tentative_set,
                'list_set_value': list_set_value,
                'set_path': path if tentative_set else [],
                'stage': stage,
                'original': original,
            })
    return operations


def _promote(value):
    """Turn a concrete container into symbolic sparse state."""
    if _mapping(value):
        result = SparseDict()
        for key, child in value.items():
            cell = Cell(initial=deepcopy(child), current=deepcopy(child))
            result.initial[key] = cell
            result.current[key] = cell
        return result
    if _sequence(value):
        result = SparseList()
        for child in value:
            cell = Cell(initial=deepcopy(child), current=deepcopy(child))
            result.initial.append(cell)
            result.current.append(cell)
        return result
    if _set(value):
        result = SparseSet()
        for child in value:
            result.initial[child] = True
            result.current[child] = True
        return result
    return value


def _matches(symbolic, concrete):
    """Check a concrete old value against known symbolic state."""
    if symbolic is UNKNOWN:
        return True
    if symbolic is ABSENT:
        return False
    if isinstance(symbolic, SparseDict):
        if not _mapping(concrete):
            return False
        for key, cell in symbolic.current.items():
            if cell.current is ABSENT:
                if key in concrete:
                    return False
            elif key not in concrete or not _matches(
                    cell.current, concrete[key]):
                return False
        return True
    if isinstance(symbolic, SparseList):
        if not _sequence(concrete) or len(symbolic.current) != len(concrete):
            return False
        return all(_matches(cell.current, value)
                   for cell, value in zip(symbolic.current, concrete))
    if isinstance(symbolic, SparseSet):
        if not _set(concrete):
            return False
        for item, present in symbolic.current.items():
            if (item in concrete) != present:
                return False
        return True
    return _same(symbolic, concrete)


def _materialize(value, initial=False):
    attribute = 'initial' if initial else 'current'
    if isinstance(value, SparseDict):
        mapping = value.initial if initial else value.current
        return {
            key: _materialize(getattr(cell, attribute), initial)
            for key, cell in mapping.items()
            if getattr(cell, attribute) is not ABSENT
        }
    if isinstance(value, SparseList):
        sequence = value.initial if initial else value.current
        return [
            _materialize(getattr(cell, attribute), initial)
            for cell in sequence
            if getattr(cell, attribute) is not ABSENT
        ]
    if isinstance(value, SparseSet):
        mapping = value.initial if initial else value.current
        return {item for item, present in mapping.items() if present}
    return value


class Composer(object):
    """Apply patches symbolically and preserve A values needed for revert."""

    def __init__(self):
        self.root = Cell()

    def compose(self, first_patch, second_patch):
        for operation in _normalize(first_patch, 1):
            self._apply(operation)
        for operation in _normalize(second_patch, 2):
            self._apply(operation)

    def _apply(self, op):
        if op['tentative_set'] and not op['parent_path']:
            self._set_operation(self.root, op)
            return

        if not op['path'] and not op['tentative_set']:
            if op['tentative_set']:
                self._set_operation(self.root, op)
            else:
                raise _error('Root-level replacement cannot be composed.',
                             op, op['stage'])
            return

        if self.root.current is UNKNOWN:
            self._infer_root(op)
        parent = self._descend(self.root, op)
        if op['tentative_set'] and (
                self._is_set_parent(parent) or not op['parent_path']):
            self._set_operation(parent, op)
        else:
            self._child_operation(parent, op)

    @staticmethod
    def _is_set_parent(parent):
        return isinstance(parent, Cell) and isinstance(
            parent.current, SparseSet)

    def _infer_root(self, op):
        if op['tentative_set'] and not op['parent_path']:
            sparse = SparseSet()
        elif isinstance(op['path'][0], int) and not isinstance(
                op['path'][0], bool):
            sparse = SparseList()
        else:
            sparse = SparseDict()
        self.root.initial = sparse
        self.root.current = sparse

    def _prepare_container(self, cell, op):
        if cell.current is ABSENT:
            raise _error('Parent path was removed before this operation.',
                         op, op['stage'])
        if isinstance(cell.current,
                      (SparseDict, SparseList, SparseSet)):
            return
        if _container(cell.current):
            promoted = _promote(cell.current)
            cell.current = promoted
            if cell.initial is ABSENT:
                cell.initial = ABSENT
                return
            if _container(cell.initial) and (
                    (_mapping(cell.initial) and isinstance(
                        promoted, SparseDict)) or
                    (_sequence(cell.initial) and isinstance(
                        promoted, SparseList)) or
                    (_set(cell.initial) and isinstance(
                        promoted, SparseSet))):
                cell.initial = _promote(cell.initial)
            return
        if cell.current is UNKNOWN:
            return
        raise _error('Path descends through a non-container value.',
                     op, op['stage'])

    def _descend(self, root_cell, op):
        cell = root_cell
        path = op['parent_path']
        for position, key in enumerate(path):
            self._prepare_container(cell, op)
            container = cell.current

            if container is UNKNOWN:
                if op['tentative_set'] and position + 1 == len(path):
                    container = SparseSet()
                    cell.initial = SparseSet()
                    cell.current = container
                elif isinstance(key, int) and not isinstance(key, bool):
                    child = self._promote_unknown_list_parent(
                        cell, key, op)
                    if position + 1 < len(path):
                        cell = child
                        continue
                else:
                    container = SparseDict()
                    cell.initial = container
                    cell.current = container

            if isinstance(container, SparseDict):
                try:
                    child = container.current[key]
                except KeyError:
                    child = Cell()
                    container.current[key] = child
                container.initial.setdefault(key, child)
                if child.current is UNKNOWN and \
                        position + 1 == len(path):
                    next_key = op['key']
                    if isinstance(next_key, int) and not isinstance(
                            next_key, bool):
                        if op['tentative_set']:
                            inferred = SparseSet()
                            child.current = inferred
                            child.initial = SparseSet()
                        else:
                            self._fill_unknown_sequence(child, op)
                        if position + 1 < len(path):
                            cell = child
                            continue
                        return child
                    elif not op['tentative_set']:
                        inferred = SparseDict()
                        child.current = inferred
                        child.initial = inferred
                        if position + 1 == len(path):
                            return child
                    else:
                        inferred = SparseSet()
                        child.current = inferred
                        child.initial = SparseSet()
                        if position + 1 == len(path):
                            return child
                elif child.current is UNKNOWN and \
                        position + 1 < len(path):
                    next_key = path[position + 1]
                    if isinstance(next_key, int) and not isinstance(
                            next_key, bool):
                        inferred = SparseList()
                        child.current = inferred
                        child.initial = inferred
                    elif not op['tentative_set']:
                        inferred = SparseDict()
                        child.current = inferred
                        child.initial = inferred
                elif child.current is not UNKNOWN and \
                        child.current is not ABSENT and \
                        position + 1 == len(path) and \
                        _container(child.current):
                    self._prepare_container(child, op)
                    return child
            elif isinstance(container, SparseList):
                if not isinstance(key, int) or isinstance(key, bool):
                    raise _error('List indices must be integers.',
                                 op, op['stage'])
                if key < 0:
                    raise _error('List parent index is out of range.',
                                 op, op['stage'])
                while len(container.current) <= key:
                    anchor = Cell()
                    container.current.append(anchor)
                    container.initial.append(anchor)
                child = container.current[key]
                if child.current is UNKNOWN and \
                        position + 1 == len(path) and \
                        op['tentative_set'] and key == op['key']:
                    inferred = SparseSet()
                    child.current = inferred
                    child.initial = SparseSet()
                    return child
                elif child.current is UNKNOWN and \
                        position + 1 == len(path) and \
                        not op['tentative_set']:
                    next_key = op['key']
                    if isinstance(next_key, int) and not isinstance(
                            next_key, bool):
                        inferred = SparseList()
                        child.current = inferred
                        child.initial = inferred
                    else:
                        inferred = SparseDict()
                        child.current = inferred
                        child.initial = inferred
                        return child
                elif child.current is not UNKNOWN and \
                        child.current is not ABSENT and \
                        position + 1 == len(path) and \
                        _container(child.current):
                    self._prepare_container(child, op)
                    return child
                elif child.current is UNKNOWN and \
                        position + 1 < len(path):
                    next_key = path[position + 1]
                    if isinstance(next_key, int) and not isinstance(
                            next_key, bool):
                        inferred = SparseList()
                        child.current = inferred
                        child.initial = inferred
                    elif not op['tentative_set']:
                        inferred = SparseDict()
                        child.current = inferred
                        child.initial = inferred
            elif isinstance(container, SparseSet):
                raise _error('Set members cannot contain subpaths.',
                             op, op['stage'])
            else:
                raise _error('Path descends through a non-container value.',
                             op, op['stage'])
            cell = child
        return cell

    def _fill_unknown_sequence(self, cell, op):
        old_value = op['old'] if op['action'] == CHANGE else op['value']
        index = op['key']
        if index < 0:
            raise _error('List parent index is out of range.',
                         op, op['stage'])
        if _sequence(old_value):
            promoted = _promote(deepcopy(old_value))
        else:
            promoted = SparseList()
            while len(promoted.current) <= index:
                anchor = Cell()
                promoted.current.append(anchor)
                promoted.initial.append(anchor)
            promoted.current[index] = Cell(
                initial=deepcopy(old_value),
                current=deepcopy(old_value))
            promoted.initial[index] = promoted.current[index]
        cell.current = promoted
        cell.initial = promoted
        return promoted.current[index]

    def _promote_unknown_list_parent(self, cell, key, op):
        old_container = None
        if op['action'] == CHANGE:
            old_container = op['old']
        elif op['action'] == REMOVE:
            old_container = op['value']

        if old_container is None or not _sequence(old_container):
            raise _error('Cannot infer unknown list parent without an old '
                         'sequence value.', op, op['stage'])
        if key < 0 or key >= len(old_container):
            raise _error('List parent index is out of range.',
                         op, op['stage'])
        promoted = _promote(deepcopy(old_container))
        if key < 0 or key >= len(promoted.current):
            raise _error('List parent index is out of range.',
                         op, op['stage'])
        cell.current = promoted
        cell.initial = promoted
        child = promoted.current[key]
        return child

    def _set_operation(self, cell, op):
        if cell.current is UNKNOWN:
            cell.current = SparseSet()
            cell.initial = SparseSet()
        elif _set(cell.current):
            promoted = _promote(cell.current)
            cell.current = promoted
        if not isinstance(cell.current, SparseSet):
            raise _error('Set operation used on a non-set path.',
                         op, op['stage'])
        initial_mapping = cell.initial.current if isinstance(
            cell.initial, SparseSet) else None

        for item in op['value']:
            try:
                hash(item)
            except TypeError:
                raise _error('Set members must be hashable.',
                             op, op['stage'])
            tracked = item in cell.current.current
            present = cell.current.current.get(item, False)
            if op['action'] == ADD:
                if present:
                    raise _error('Set member already exists.',
                                 op, op['stage'])
                cell.current.current[item] = True
                if initial_mapping is not None:
                    initial_mapping.setdefault(item, False)
            else:
                if tracked and not present:
                    raise _error('Set member does not exist.',
                                 op, op['stage'])
                if not tracked:
                    cell.current.current[item] = True
                cell.current.current[item] = False
                if initial_mapping is not None:
                    initial_mapping.setdefault(item, True)

    def _child_operation(self, parent_cell, op):
        parent = parent_cell.current
        if isinstance(parent, SparseDict):
            self._dict_operation(parent, op)
        elif isinstance(parent, SparseList):
            self._list_operation(parent, op)
        else:
            raise _error('Operation path is not inside a dictionary or '
                         'list.', op, op['stage'])

    def _dict_operation(self, sparse, op):
        key = op['key']
        try:
            hash(key)
        except TypeError:
            raise _error('Dictionary keys must be hashable.',
                         op, op['stage'])
        cell = sparse.current.setdefault(key, Cell())

        if op['action'] == ADD:
            if cell.current is not UNKNOWN and cell.current is not ABSENT:
                raise _error('Dictionary key already exists.',
                             op, op['stage'])
            cell.current = deepcopy(op['value'])
            if op['stage'] == 2:
                if key not in sparse.initial and cell.initial is UNKNOWN:
                    cell.initial = ABSENT
            else:
                cell.initial = ABSENT
        elif op['action'] == REMOVE:
            if cell.current is ABSENT:
                raise _error('Dictionary key does not exist.',
                             op, op['stage'])
            if cell.current is not UNKNOWN and not _matches(
                    cell.current, op['value']):
                raise _error('Dictionary remove old value does not match.',
                             op, op['stage'])
            cell.current = ABSENT
            if cell.initial is ABSENT:
                sparse.initial.pop(key, None)
                cell.initial = ABSENT
            else:
                sparse.initial[key] = cell
                if cell.initial is UNKNOWN:
                    cell.initial = deepcopy(op['value'])
                elif op['stage'] == 1:
                    cell.initial = deepcopy(op['value'])
        else:
            if op['stage'] == 1:
                sparse.initial.setdefault(key, cell)
            else:
                sparse.initial.setdefault(key, cell)
            self._change_cell(cell, op)

    def _list_operation(self, sparse, op):
        index = op['key']
        if not isinstance(index, int) or isinstance(index, bool):
            raise _error('List indices must be integers.',
                         op, op['stage'])

        if op['action'] == ADD:
            restored = self._restored_deletion(sparse, index, op)
            if restored is None and (
                    index < 0 or
                    index > len(sparse.current)):
                raise _error('List insert index is out of range.',
                             op, op['stage'])
            if restored is not None:
                restored.current = deepcopy(op['value'])
                if restored.initial is UNKNOWN:
                    restored.initial = deepcopy(op['value'])
                sparse.current.insert(index, restored)
            else:
                cell = Cell(initial=ABSENT, current=deepcopy(op['value']))
                sparse.current.insert(index, cell)
            return

        if index < 0:
            raise _error('List index is out of range.', op, op['stage'])
        if index > len(sparse.current):
            if op['action'] == CHANGE and op['stage'] == 1:
                while len(sparse.current) < index:
                    anchor = Cell()
                    sparse.current.append(anchor)
                    sparse.initial.append(anchor)
                cell = Cell(initial=deepcopy(op['old']),
                            current=deepcopy(op['old']))
                sparse.current.append(cell)
                sparse.initial.append(cell)
            else:
                raise _error('List index is out of range.',
                             op, op['stage'])
        while len(sparse.current) < index:
            anchor = Cell()
            sparse.current.append(anchor)
            sparse.initial.append(anchor)
        if index == len(sparse.current):
            if op['action'] == REMOVE:
                cell = Cell(initial=deepcopy(op['value']),
                            current=deepcopy(op['value']))
                sparse.current.append(cell)
                sparse.initial.append(cell)
            elif op['action'] == CHANGE and op['stage'] == 1:
                cell = Cell(initial=deepcopy(op['old']),
                            current=deepcopy(op['old']))
                sparse.current.append(cell)
                sparse.initial.append(cell)
            else:
                raise _error('List index is out of range.',
                             op, op['stage'])
        cell = sparse.current[index]

        if cell.current is UNKNOWN and op['action'] == CHANGE:
            cell.current = deepcopy(op['old'])
            if cell.initial is UNKNOWN:
                cell.initial = deepcopy(op['old'])

        if op['action'] == REMOVE:
            if cell.current is ABSENT or cell.current is UNKNOWN:
                raise _error('List position does not exist.',
                             op, op['stage'])
            if _different(cell.current, op['value']):
                raise _error('List remove old value does not match.',
                             op, op['stage'])
            sparse.current.pop(index)
            if op['stage'] == 1:
                cell.initial = deepcopy(op['value'])
            elif cell.initial is UNKNOWN:
                cell.initial = deepcopy(op['value'])
            cell.current = ABSENT
        else:
            same_kind = (
                (_mapping(op['old']) and _mapping(op['new'])) or
                (_sequence(op['old']) and _sequence(op['new'])) or
                (_set(op['old']) and _set(op['new'])))
            incompatible = (_container(op['old']) or
                            _container(op['new'])) and not same_kind
            if incompatible:
                replacement = Cell(initial=ABSENT,
                                   current=deepcopy(op['new']))
                if cell not in sparse.initial:
                    sparse.initial.append(cell)
                sparse.current[index] = replacement
                cell.current = ABSENT
                if cell.initial is UNKNOWN:
                    cell.initial = deepcopy(op['old'])
            else:
                self._change_cell(cell, op)

    def _restored_deletion(self, sparse, insert_index, op):
        candidates = [
            cell for cell in sparse.initial
            if cell.current is ABSENT and cell.initial is not ABSENT
        ]
        if not candidates:
            return None
        for cell in candidates:
            before = self._initial_neighbors(sparse, cell)
            if before == insert_index and _same(
                    _materialize(cell.initial, initial=True),
                    op['value']):
                return cell
        return None

    @staticmethod
    def _initial_neighbors(sparse, target):
        count = 0
        for cell in sparse.initial:
            if cell is target:
                return count
            if cell.current is not ABSENT and cell.initial is not ABSENT:
                count += 1
        return count

    def _change_cell(self, cell, op):
        if cell.current is ABSENT:
            raise _error('Change requires an existing value.',
                         op, op['stage'])
        same_container_kind = (
            (_mapping(op['old']) and _mapping(op['new'])) or
            (_sequence(op['old']) and _sequence(op['new'])) or
            (_set(op['old']) and _set(op['new'])))
        if _container(op['old']) and same_container_kind:
            if not _matches(cell.current, op['old']):
                raise _error('Change old value does not match.',
                             op, op['stage'])
            promoted = _promote(deepcopy(op['old']))
            if cell.initial is UNKNOWN and not _container(cell.initial):
                cell.initial = deepcopy(op['old'])
            cell.current = promoted
            return

        if cell.current is not UNKNOWN and _different(
                cell.current, op['old']):
            raise _error('Change old value does not match.',
                         op, op['stage'])
        if cell.initial is UNKNOWN:
            cell.initial = deepcopy(op['old'])
        cell.current = deepcopy(op['new'])


def _node(path, dot_notation):
    if not path:
        return ''
    if dot_notation and all(
            isinstance(part, str) and '.' not in part for part in path):
        return '.'.join(path)
    return list(path)


def _list_matches(initial_cells, current_cells):
    """Match list cells by their tracked identity."""
    initial_by_id = {id(cell): index for index, cell in
                     enumerate(initial_cells)}
    result = []
    used_initial = set()
    used_current = set()
    for right, cell in enumerate(current_cells):
        identity = initial_by_id.get(id(cell))
        if identity is not None and identity not in used_initial:
            result.append((identity, right))
            used_initial.add(identity)
            used_current.add(right)
    return sorted(result)


def _serialize_cell(cell, path, dot_notation):
    initial = cell.initial
    current = cell.current

    if initial is ABSENT and isinstance(
            current, (SparseDict, SparseList, SparseSet)):
        return [(ADD, _node(path[:-1], dot_notation),
                 [(path[-1], _materialize(current))])]
    if current is ABSENT and isinstance(
            initial, (SparseDict, SparseList, SparseSet)):
        return [(REMOVE, _node(path[:-1], dot_notation),
                 [(path[-1], _materialize(initial, initial=True))])]

    if initial is UNKNOWN:
        initial = deepcopy(current)
    if isinstance(initial, SparseDict) and isinstance(current, SparseDict):
        return _serialize_dict(initial, current, path, dot_notation)
    if isinstance(initial, SparseList) and isinstance(current, SparseList):
        return _serialize_list(initial, current, path, dot_notation)
    if isinstance(initial, SparseSet) and isinstance(current, SparseSet):
        return _serialize_set(initial, current, path, dot_notation)

    sparse_types = (SparseDict, SparseList, SparseSet)
    if isinstance(initial, sparse_types) or isinstance(current, sparse_types):
        old_value = _materialize(initial, initial=True)
        new_value = _materialize(current)
        if _different(old_value, new_value):
            return [(CHANGE, _node(path, dot_notation),
                     (old_value, new_value))]
        return []

    if initial is not ABSENT and current is ABSENT:
        if not path:
            return []
        return [(REMOVE, _node(path[:-1], dot_notation),
                 [(path[-1], _materialize(initial, initial=True))])]
    if initial is ABSENT and current is not ABSENT:
        if not path:
            return []
        return [(ADD, _node(path[:-1], dot_notation),
                 [(path[-1], _materialize(current))])]
    if initial is ABSENT and current is ABSENT:
        return []
    old_value = _materialize(initial, initial=True)
    new_value = _materialize(current)
    if _different(old_value, new_value):
        return [(CHANGE, _node(path, dot_notation),
                 (old_value, new_value))]
    return []


def _serialize_dict(initial, current, path, dot_notation):
    result = []
    initial_map = initial.initial
    current_map = current.current

    for key in initial_map.keys() & current_map.keys():
        if current_map[key].current is ABSENT and \
                initial_map[key].initial is not ABSENT:
            result.append((REMOVE, _node(path, dot_notation),
                           [(key, _materialize(
                               initial_map[key].initial,
                               initial=True))]))
            continue
        if current_map[key].current is ABSENT and \
                initial_map[key].initial is ABSENT:
            continue
        result.extend(_serialize_cell(
            Cell(initial=initial_map[key].initial,
                 current=current_map[key].current),
            path + [key], dot_notation))

    for key in sorted(current_map.keys() - initial_map.keys(), key=repr):
        cell = current_map[key]
        if cell.current is not ABSENT:
            result.append((ADD, _node(path, dot_notation),
                           [(key, _materialize(cell.current))]))

    for key in sorted(initial_map.keys() - current_map.keys(),
                      key=repr, reverse=True):
        cell = initial_map[key]
        if cell.initial is not ABSENT:
            result.append((REMOVE, _node(path, dot_notation),
                           [(key, _materialize(cell.initial,
                                               initial=True))]))
    return result


def _serialize_list(initial, current, path, dot_notation):
    result = []
    initial_cells = [
        (index, cell) for index, cell in enumerate(initial.initial)
                     if cell.initial is not ABSENT and
                     cell.initial is not UNKNOWN]
    current_live = [(index, cell) for index, cell in enumerate(current.current)
                    if cell.current is not ABSENT]
    current_live = [
        (index, cell) for index, cell in current_live
        if cell.initial is not UNKNOWN or
        isinstance(cell.current,
                   (SparseDict, SparseList, SparseSet))]
    known_current = [(index, cell) for index, cell in current_live
                     if cell.initial is not ABSENT and
                     cell.initial is not UNKNOWN]
    matches = _list_matches([cell for _, cell in initial_cells],
                            [cell for _, cell in known_current])
    matched_initial = {left for left, _ in matches}
    matched_current = {right for _, right in matches}
    structural_matches = set()

    for left, right in matches:
        initial_index, _ = initial_cells[left]
        _, current_cell = known_current[right]
        old_value = initial_cells[left][1].initial
        new_value = current_cell.current
        if isinstance(old_value,
                      (SparseDict, SparseList, SparseSet)) or \
                isinstance(new_value,
                           (SparseDict, SparseList, SparseSet)):
            structural_matches.add((left, right))
            continue
        result.extend(_serialize_cell(
            Cell(initial=old_value, current=new_value),
            path + [initial_index], dot_notation))

    tombstones = [
        (index, cell) for index, cell in enumerate(initial.initial)
        if cell.initial is not ABSENT and
        cell.initial is not UNKNOWN and
        cell.current is ABSENT
    ]
    for left, cell in tombstones:
        if any(initial_index == left
               for initial_index, _ in
               (initial_cells[match[0]] for match in matches)):
            continue
        structural_lefts = {match[0] for match in structural_matches}
        if left not in matched_initial or left in structural_lefts:
            result.append((REMOVE, _node(path, dot_notation),
                           [(left, _materialize(cell.initial,
                                                initial=True))]))

    for right, (current_index, cell) in reversed(
            list(enumerate(current_live))):
        known_position = next(
            (position for position, (_, candidate)
             in enumerate(known_current) if candidate is cell),
            None)
        structural_rights = {match[1] for match in structural_matches}
        if known_position is None or known_position not in \
                matched_current or known_position in structural_rights:
            result.append((ADD, _node(path, dot_notation),
                           [(current_index,
                             _materialize(cell.current))]))
    return result


def _serialize_set(initial, current, path, dot_notation):
    result = []
    removed = set()
    added = set()
    for item in set(initial.current) | set(current.current):
        old_present = initial.current.get(item, False)
        new_present = current.current.get(item, False)
        if old_present is UNKNOWN:
            old_present = new_present
        if old_present and not new_present:
            removed.add(item)
        elif new_present and not old_present:
            added.add(item)
    if removed:
        result.append((REMOVE, _node(path, dot_notation), [(0, removed)]))
    if added:
        result.append((ADD, _node(path, dot_notation), [(0, added)]))
    return result


def compose(first_patch, second_patch, dot_notation=True):
    """Compose consecutive ``A→B`` and ``B→C`` patches into ``A→C``.

    :param first_patch: Iterable of dictdiffer tuples transforming A to B.
    :param second_patch: Iterable of dictdiffer tuples transforming B to C.
    :param dot_notation: Emit dotted paths when all keys are dot-free strings.
    :raises PatchCompositionError: if the patches have incompatible structure
        or contradictory old values/preconditions.
    :returns: A list of standard dictdiffer tuples transforming A to C.
    """
    composer = Composer()
    composer.compose(list(first_patch), list(second_patch))
    result = _serialize_cell(composer.root, [], dot_notation)
    order = {CHANGE: 0, REMOVE: 1, ADD: 2}
    result.sort(key=lambda item: order[item[0]])

    groups = {}
    for index, item in enumerate(result):
        if item[0] != ADD or not item[2]:
            continue
        key, value = item[2][0]
        node = item[1]
        parent_path = node.split('.') if isinstance(node, str) \
            else list(node)
        root_set_value = not parent_path and key == 0 and \
            isinstance(value, MutableSet)
        if isinstance(key, int) and not isinstance(key, bool) and \
                not root_set_value:
            groups.setdefault(item[1], []).append(index)

    for indexes in groups.values():
        items = [result[index] for index in indexes]
        items.sort(key=lambda item: item[2][0][0])
        for index, item in zip(indexes, items):
            result[index] = item
    return result
