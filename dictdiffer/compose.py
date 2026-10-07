# SPDX-FileCopyrightText: 2026
# SPDX-License-Identifier: MIT

"""Compose two sequential dictdiffer patches into one net patch.

Given ``patch1`` transforming *A -> B* and ``patch2`` transforming *B -> C*,
:func:`compose` returns a single patch transforming *A -> C*. Neither the
original documents nor the intermediate document *B* have to be supplied; all
required values are taken from the patches themselves (old values carried by
``change``/``remove`` items, new values carried by ``add``/``change`` items).

Supported input
---------------
* Patches produced by :func:`dictdiffer.diff`, both in the default grouped
  form (``('add', 'a', [(0, 1), (1, 2)])``) and the expanded form
  (``expand=True``).
* Dot-notation (string) and list/tuple node paths, possibly mixed within one
  patch.
* Dictionary sub-paths (``add``/``remove``/``change`` at any depth), lists
  (insert/delete with compensation of index drift) and ``set`` containers.
* Patches created with ``path_limit``: wholesale subtree replacement is folded
  by symbolically applying the inner patch to the value embedded in the outer
  patch.

Operations that cancel over the two phases (add-then-delete, or
change-then-revert) are dropped from the net patch. The resulting patch keeps
the original *A* values on every ``change``/``remove`` item, so it can be
reverted with :func:`dictdiffer.revert`.

Conflicts
---------
Composition is purely symbolic - the patches are never applied to caller data.
When the preconditions stated by ``patch2`` are incompatible with the state
produced by ``patch1`` (e.g. ``patch2`` changes a key that ``patch1`` removes,
or expects another old value/container type), a :class:`ComposeConflict` is
raised *before* any result is returned. Sequentially applying the two patches
to incompatible data would otherwise fail halfway through (typically with a
:class:`KeyError`).

Limitations
-----------
* Integer keys of *dictionaries* cannot be told apart from list indices;
  composing a mapping that uses integer keys raises
  :class:`UncomposablePatchError`.
* The patch form ``('add', node, [(0, {set})])`` is ambiguous between a set
  mutation and inserting a ``set`` as the first element of an empty list;
  compose interprets the single-entry form as a set operation (as emitted for
  sets by ``diff``).
* Root-level wholesale changes (``('change', '', ...)``) are composed
  symbolically, but applying them is outside what :func:`dictdiffer.patch`
  supports (it cannot replace the whole passed document); the composed tuple
  is still correct for inspection or custom application.
* Hand-crafted patches that no single ``diff`` can produce (simultaneous
  insert and delete at one list node, insertion outside the trailing range)
  are rejected with :class:`UncomposablePatchError` or
  :class:`InvalidPatchError`.
"""

from collections.abc import MutableMapping, MutableSequence, MutableSet
from copy import deepcopy

from . import patch as _apply_patch
from .utils import is_super_path

__all__ = ('compose', 'ComposeError', 'ComposeConflict',
           'UncomposablePatchError', 'InvalidPatchError')

ADD = 'add'
REMOVE = 'remove'
CHANGE = 'change'
_ACTIONS = (ADD, REMOVE, CHANGE)


class ComposeError(Exception):
    """Base class for all errors raised during patch composition."""


class ComposeConflict(ComposeError):
    """The two patches state contradictory preconditions.

    Applying them sequentially to the same document cannot succeed; e.g. the
    second patch touches a node that the first patch removes, or expects a
    different old value or container type than the first patch leaves behind.
    """

    def __init__(self, message, path=None):
        if path:
            message = '{0}: {1}'.format(format_path(path), message)
        super(ComposeConflict, self).__init__(message)
        self.path = tuple(path) if path is not None else ()


class UncomposablePatchError(ComposeError):
    """The patches use a feature that composition cannot represent."""


class InvalidPatchError(ComposeError):
    """A patch tuple is malformed and cannot be produced by ``diff``."""


def format_path(path):
    """Render a node path tuple for error messages."""
    if not path:
        return '<root>'
    if all(isinstance(part, str) for part in path):
        return '.'.join(str(part) for part in path)
    return repr(list(path))


# -- Parsing ----------------------------------------------------------------

class _Node(object):
    """All patch items touching one node path of one phase."""

    def __init__(self):
        self.adds = {}
        self.removes = {}
        self.change = None  # (old, new)
        self.order = []  # (seq, action, key)


class _Phase(object):
    """Parsed view of one patch (one transition between documents)."""

    def __init__(self, raw):
        self.nodes = {(): _Node()}
        self.dotted = True
        seq = 0
        for item in raw:
            try:
                action, node, payload = item
            except (TypeError, ValueError):
                raise InvalidPatchError(
                    'every patch item must be a (action, node, changes) '
                    'tuple; got {0!r}'.format(item))
            if action not in _ACTIONS:
                raise InvalidPatchError(
                    'unknown patch action {0!r}; expected one of {1}'.format(
                        action, _ACTIONS))
            path = parse_node(node)
            if not isinstance(node, str):
                self.dotted = False
            record = self.nodes.setdefault(path, _Node())

            if action == CHANGE:
                if not isinstance(payload, (list, tuple)) or len(payload) != 2:
                    raise InvalidPatchError(
                        'change payload must be an (old, new) pair at node '
                        '{0}; got {1!r}'.format(format_path(path), payload))
                if record.change is not None:
                    raise InvalidPatchError(
                        'multiple change items at node {0}'.format(
                            format_path(path)))
                record.change = (deepcopy(payload[0]), deepcopy(payload[1]))
                record.order.append((seq, CHANGE, None))
                seq += 1
                continue

            if not isinstance(payload, (list, tuple)):
                raise InvalidPatchError(
                    '{0} payload at node {1} must be a list of (key, value) '
                    'pairs; got {2!r}'.format(action, format_path(path),
                                             payload))
            bucket = record.adds if action == ADD else record.removes
            for pair in payload:
                try:
                    key, value = pair
                except (TypeError, ValueError):
                    raise InvalidPatchError(
                        '{0} items must be (key, value) pairs at node {1}; '
                        'got {2!r}'.format(action, format_path(path), pair))
                if key in bucket:
                    raise InvalidPatchError(
                        '{0} lists key {1!r} twice at node {2}'.format(
                            action, key, format_path(path)))
                bucket[key] = deepcopy(value)
                record.order.append((seq, action, key))
                seq += 1


def parse_node(node):
    """Turn a string/list patch node into a path tuple."""
    if node == '' or node is None or node == []:
        return ()
    if isinstance(node, str):
        return tuple(node.split('.'))
    if isinstance(node, (list, tuple)):
        return tuple(node)
    raise InvalidPatchError(
        'patch node must be a string or list path; got {0!r}'.format(node))


# -- Symbolic application ----------------------------------------------------

def subtree_tuples(phase, path, include_self_entries=True):
    """Return interior patch items of *phase* at/under *path*.

    Paths are rewritten relative to *path* (it becomes the empty root). A
    wholesale ``change`` record of *path* is always excluded. By default the
    node's own add/remove entries are interior operations on the subtree;
    pass ``include_self_entries=False`` when folding edits of a persistent
    node whose boundary is handled elsewhere (only strict descendants).
    """
    records = []
    for other, record in phase.nodes.items():
        if not is_super_path(path, other):
            continue
        if not record.order:
            continue
        records.append((min(seq for seq, _, _ in record.order),
                        other, record))
    result = []
    for _, other, record in sorted(records, key=lambda item: item[0]):
        if other == path and record.change is not None:
            continue  # wholesale boundary value, not an interior delta
        if other == path and not include_self_entries:
            continue
        rel = list(other[len(path):])
        node = rel if rel else ''
        emitted = {ADD: None, REMOVE: None}
        for _, action, key in record.order:
            if action == CHANGE:
                result.append((CHANGE, node, record.change))
            else:
                if other == path and not include_self_entries:
                    continue
                bucket = record.adds if action == ADD else record.removes
                if emitted[action] is None:
                    emitted[action] = []
                emitted[action].append((key, bucket[key]))
        for action in (ADD, REMOVE):
            if emitted[action]:
                result.append((action, node, emitted[action]))
    return result


def _safe_lookup(value, rel):
    cur = value
    for part in rel:
        cur = cur[part] if not isinstance(cur, MutableSequence) else cur[int(part)]
    return cur


def symbolic_apply(tuples, value, path):
    """Apply *tuples* on a deep-copied concrete value.

    Any failure means the two patches contradict each other; it is reported as
    a ComposeConflict instead of escaping as a mid-patch KeyError.
    """
    target = deepcopy(value)
    try:
        return _apply_patch(iter(tuples), target)
    except ComposeError:
        raise
    except Exception as exc:
        raise ComposeConflict(
            'patch does not apply to the state produced by the previous '
            'patch ({0}: {1})'.format(type(exc).__name__, exc), path)


# -- Node states and result algebra -----------------------------------------

# State s of a node in one phase:
#   None                         persistent (or absent with no boundary)
#   ('A', value)                 added by this phase (new value)
#   ('R', old)                   removed by this phase (old value)
#   (CHANGE, old, new)              wholesale changed by this phase


class _R(object):
    """Net result for one node; built by :meth:`Composer.compose_at`."""

    __slots__ = ('kind', 'value', 'old', 'new', 'children', 'appends',
                 'removes', 'set_add', 'set_remove')

    def __init__(self, kind):
        self.kind = kind
        self.value = None       # for 'add'
        self.old = None         # for 'remove' / 'change'
        self.new = None         # for 'change'
        self.children = {}      # dict stay: key -> _R; list stay: idx -> _R
        self.appends = []       # list stay: [(index, value)]
        self.removes = []       # list stay: [(index, old)]
        self.set_add = None     # set stay: set
        self.set_remove = None

    @staticmethod
    def add(value):
        result = _R('add')
        result.value = value
        return result

    @staticmethod
    def remove(old):
        result = _R('remove')
        result.old = old
        return result

    @staticmethod
    def change(old, new):
        result = _R('change')
        result.old = old
        result.new = new
        return result

    @staticmethod
    def stay(kind):
        return _R(kind)


# -- Composer ---------------------------------------------------------------

def _state_of(phase, path):
    """State a phase assigns to node *path* via its parent boundary.

    Returns ``None`` (persistent/absent), ``('A', value)``, ``('R', old)``
    or ``(CHANGE, old, new)``.
    """
    if not path:
        return None
    parent = phase.nodes.get(path[:-1])
    if parent is None:
        return None
    key = path[-1]
    if key in parent.adds:
        return (ADD, parent.adds[key])
    if key in parent.removes:
        return (REMOVE, parent.removes[key])
    if parent.change is not None:
        return (CHANGE,) + parent.change
    return None


def _record(phase, path):
    return phase.nodes.get(path)


def _entries_kind(phase, path):
    """Container kind implied by add/remove entry keys at one node.

    Returns ``(kind, ambiguous)`` where *ambiguous* marks the single
    ``(0, set)`` form that could equally be a set mutation or a list
    appending one set.
    """
    record = phase.nodes.get(path)
    if record is None or (not record.adds and not record.removes):
        return None, False
    keys = list(record.adds) + list(record.removes)
    if all(isinstance(k, int) and not isinstance(k, bool) for k in keys):
        if len(keys) == 1 and keys[0] == 0 and isinstance(
                record.adds.get(0, record.removes.get(0)), MutableSet):
            return 'set', True
        return 'list', False
    if all(not isinstance(k, int) or isinstance(k, bool) for k in keys):
        return 'dict', False
    raise UncomposablePatchError(
        'mixed integer and non-integer keys at {0}; dictionaries with '
        'integer keys are not supported by compose'.format(format_path(path)))


def _value_kind(value):
    if isinstance(value, MutableSet):
        return 'set'
    if isinstance(value, MutableMapping):
        return 'dict'
    if isinstance(value, MutableSequence):
        return 'list'
    return None


class Composer(object):
    """Builds the symbolic A -> C patch from two parsed phases."""

    def __init__(self, phase1, phase2):
        self.p1 = phase1
        self.p2 = phase2

    def compose(self):
        self._validate_orphans()
        root = self.compose_at(())
        if root is None:
            return []
        if root.kind in ('dict', 'list', 'set'):
            tuples = []
            self._emit(root, (), tuples)
            return tuples
        if root.kind == 'add':
            return [(ADD, '', [(0, root.value)])]
        if root.kind == 'remove':
            return [(REMOVE, '', [(0, root.old)])]
        return [(CHANGE, '', (root.old, root.new))]

    # -- orphan precondition check ------------------------------------------

    def _validate_orphans(self):
        """Nodes touched by phase 2 must be available in the B state."""
        for path in self.p2.nodes:
            if not path:
                continue
            state2 = _state_of(self.p2, path)
            if state2 is not None and state2[0] == ADD:
                continue  # p2 itself creates the node
            before = _state_of(self.p1, path)
            if before is None:
                continue  # persistent node; diff guarantees the precondition
            if before[0] == REMOVE:
                raise ComposeConflict(
                    'second patch touches {0}, removed by the first patch'
                    .format(format_path(path)), path)
            if before[0] == CHANGE and not _value_kind(before[2]):
                if self._has_children(self.p2, path):
                    raise ComposeConflict(
                        'second patch edits below {0}, replaced by a scalar '
                        'in the first patch'.format(format_path(path)), path)

    # -- per-node recursion --------------------------------------------------

    @staticmethod
    def _has_children(phase, path):
        return any(other != path and is_super_path(path, other)
                   for other in phase.nodes)

    def compose_at(self, path, s1=None, s2=None):
        t1 = _state_of(self.p1, path) if s1 is None else s1
        t2 = _state_of(self.p2, path) if s2 is None else s2
        r1, r2 = _record(self.p1, path), _record(self.p2, path)
        if t1 is None and r1 is not None and r1.change is not None:
            t1 = (CHANGE,) + r1.change
        if t2 is None and r2 is not None and r2.change is not None:
            t2 = (CHANGE,) + r2.change

        if t1 is None and t2 is None and not (
                self._has_children(self.p1, path)
                or self._has_children(self.p2, path)
                or (r1 and (r1.adds or r1.removes))
                or (r2 and (r2.adds or r2.removes))):
            return None

        if t1 is not None or t2 is not None:
            return self._compose_boundary(path, t1, t2)

        kind = self._persistent_kind(path)
        if kind == 'set':
            return self._compose_set(path)
        if kind == 'dict':
            return self._compose_dict(path)
        return self._compose_list(path)

    def _persistent_kind(self, path):
        kinds = set()
        ambiguous = set()
        for phase in (self.p1, self.p2):
            kind, unsure = _entries_kind(phase, path)
            if kind:
                (ambiguous if unsure else kinds).add(kind)
            direct = {other[len(path)] for other in phase.nodes
                      if other != path and is_super_path(path, other)}
            if direct:
                idx = next(iter(direct))
                kinds.add('list' if isinstance(idx, int)
                          and not isinstance(idx, bool) else 'dict')
        # A definite list/dict/set evidence from either phase resolves the
        # single-(0, set) ambiguity from the other phase.
        if 'list' in kinds:
            ambiguous.discard('set')
        if 'set' in kinds:
            ambiguous.discard('list')
        kinds |= ambiguous
        if len(kinds) > 1:
            raise ComposeConflict(
                'the patches disagree on the container type at {0} ({1})'
                .format(format_path(path), ' vs '.join(sorted(kinds))), path)
        return kinds.pop() if kinds else 'dict'

    # -- wholesale boundary combinations -----------------------------------

    def _compose_boundary(self, path, t1, t2):
        tag1 = t1[0] if t1 is not None else None
        tag2 = t2[0] if t2 is not None else None

        if tag1 is None and tag2 == ADD:
            return _R.add(self._fold(path, t1, t2))
        if tag1 is None and tag2 == REMOVE:
            # Persistent node (possibly emptied by interior p1 edits) that
            # p2 removes wholesale: fold p1 interior into the old value.
            return _R.remove(self._fold(path, t1, t2, base='old'))
        if tag1 is None and tag2 == CHANGE:
            # p1 may have edited the interior of this persistent node; the
            # change's old value must be the A state, rebuilt by reverting
            # those interior tuples onto the B value carried by p2.
            return _R.change(self._revert_p1(path, t2[1]),
                             self._fold(path, t1, t2))

        if tag1 == ADD and tag2 is None:
            return _R.add(self._fold(path, t1, t2))
        if tag1 == REMOVE and tag2 is None:
            return _R.remove(t1[1])
        if tag1 == CHANGE and tag2 is None:
            return _R.change(t1[1], self._fold(path, t1, t2))

        if tag1 == ADD and tag2 == ADD:
            raise ComposeConflict(
                'second patch adds {0}, already added by the first'.format(
                    format_path(path)), path)
        if tag1 == ADD and tag2 == REMOVE:
            self._expect(path, t1[1], t2[1], 'removed')
            return None
        if tag1 == ADD and tag2 == CHANGE:
            self._expect(path, t1[1], t2[1], 'changed')
            return _R.add(self._fold(path, t1, t2))

        if tag1 == REMOVE and tag2 == ADD:
            return _R.change(t1[1], self._fold(path, t1, t2))
        if tag1 == REMOVE and tag2 in (REMOVE, CHANGE):
            raise ComposeConflict(
                'second patch {0} {1}, removed by the first'.format(
                    'removes' if tag2 == REMOVE else 'changes',
                    format_path(path)), path)

        if tag1 == CHANGE and tag2 == ADD:
            raise ComposeConflict(
                'second patch adds {0}, wholesale changed by the first'
                .format(format_path(path)), path)
        if tag1 == CHANGE and tag2 == REMOVE:
            self._expect(path, t1[2], t2[1], 'removed')
            return _R.remove(self._revert_p1(path, t1[1]))
        if tag1 == CHANGE and tag2 == CHANGE:
            self._expect(path, t1[2], t2[1], 'changed')
            value = self._fold(path, t1, t2)
            if t1[1] == value:
                return None
            return _R.change(t1[1], value)
        raise ComposeConflict('unsupported boundary combination', path)

    @staticmethod
    def _expect(path, actual, expected, verb):
        if actual != expected:
            raise ComposeConflict(
                'value {0!r} {1} by the second patch is not the value '
                '{2!r} produced by the first'.format(expected, verb, actual),
                path)

    def _fold(self, path, t1, t2, base='new'):
        """Build the boundary value, folding interior tuples of both phases.

        ``base='new'`` seeds from the value after the wholesale boundary (used
        for add/change results); ``base='old'`` seeds from the value carried
        by a p2 removal, onto which the p1 interior tuples are reverted to
        reconstruct the original *A* value.
        """
        tuples1 = subtree_tuples(self.p1, path)
        tuples2 = subtree_tuples(self.p2, path)
        if base == 'old':
            # t2 is (REMOVE, b_value); reconstruct the A value by undoing all
            # of p1's edits of this subtree (boundary entries and descendants).
            return self._revert_p1(path, t2[1])
        if t2 is not None and t2[0] in (ADD, CHANGE):
            # p2 already states the complete post-boundary value; p1's
            # interior edits of the old subtree are overwritten wholesale.
            value = t2[2] if t2[0] == CHANGE else t2[1]
            tuples = tuples2
        elif t1 is not None and t1[0] in (ADD, CHANGE):
            value = t1[2] if t1[0] == CHANGE else t1[1]
            tuples = tuples1 + tuples2
        else:
            value = None
            tuples = tuples1 + tuples2
        if not tuples:
            return value
        if not isinstance(
                value, (MutableMapping, MutableSequence, MutableSet)):
            raise ComposeConflict(
                'interior patch items below scalar value at {0}'.format(
                    format_path(path)), path)
        return symbolic_apply(tuples, value, path)

    def _revert_p1(self, path, b_value):
        """Rebuild the A value at *path* from the B value p2 carries.

        Undoes p1's subtree edits in reverse global patch order. At each
        node the forward walk recurses into children after registering the
        node boundary; reverting therefore processes later/global-deeper
        items first. List index operations are undone with tail-safe
        insertion/deletion instead of swap(), which cannot order them.
        """
        value = deepcopy(b_value)
        items = []
        for other, record in self.p1.nodes.items():
            if not is_super_path(path, other):
                continue
            if not record.order:
                continue
            rel = list(other[len(path):])
            for seq, action, key in record.order:
                items.append((seq, rel, action, key, record, other))
        for _, rel, action, key, record, other in sorted(
                items, key=lambda item: -item[0]):
            if action == CHANGE:
                if not rel:
                    value = deepcopy(record.change[0])
                else:
                    parent = _safe_lookup(value, rel[:-1])
                    parent[rel[-1]] = record.change[0]
                continue
            node = value if not rel else _safe_lookup(value, rel)
            if action == ADD:
                added = record.adds[key]
                if isinstance(node, MutableSequence):
                    idx = min(key, len(node) - 1)
                    if 0 <= idx < len(node) and node[idx] == added:
                        del node[idx]
                elif isinstance(node, MutableSet) \
                        and isinstance(added, MutableSet):
                    node -= added
                else:
                    node.pop(key, None)
            elif action == REMOVE:
                old = record.removes[key]
                if isinstance(node, MutableSequence):
                    node.insert(min(key, len(node)), old)
                elif isinstance(node, MutableSet) \
                        and isinstance(old, MutableSet):
                    node |= old
                else:
                    node[key] = old
            elif action == CHANGE and rel:
                parent = _safe_lookup(value, rel[:-1])
                parent[rel[-1]] = record.change[0]
        return value

    # -- sets ---------------------------------------------------------------

    def _compose_set(self, path):
        r1, r2 = _record(self.p1, path), _record(self.p2, path)
        added1 = set(r1.adds.get(0, set()) if r1 else set())
        removed1 = set(r1.removes.get(0, set()) if r1 else set())
        added2 = set(r2.adds.get(0, set()) if r2 else set())
        removed2 = set(r2.removes.get(0, set()) if r2 else set())

        conflict = (removed2 & removed1) - added1
        if conflict:
            raise ComposeConflict(
                'second patch removes elements {0!r}, already removed by the '
                'first'.format(sorted(conflict, key=str)), path)

        net_add = (added1 | added2) - (removed1 | removed2)
        net_remove = (removed1 | removed2) - (added1 | added2)
        if not net_add and not net_remove:
            return None
        stay = _R.stay('set')
        stay.set_add, stay.set_remove = net_add, net_remove
        return stay

    # -- dictionaries -------------------------------------------------------

    def _compose_dict(self, path):
        r1, r2 = _record(self.p1, path), _record(self.p2, path)
        adds1 = r1.adds if r1 else {}
        removes1 = r1.removes if r1 else {}
        adds2 = r2.adds if r2 else {}
        removes2 = r2.removes if r2 else {}
        keys = set(adds1) | set(removes1) | set(adds2) | set(removes2)
        keys |= self._interior_keys(path)

        # A mapping that stores values under integer keys cannot be told
        # apart from a list; reject only that ambiguous boundary form.
        for bucket in (adds1, removes1, adds2, removes2):
            if any(isinstance(k, int) and not isinstance(k, bool)
                   for k in bucket):
                raise UncomposablePatchError(
                    'dictionary with integer key at {0} is not supported '
                    'by compose'.format(format_path(path)))

        stay = _R.stay('dict')
        for key in _sorted_keys(keys):
            result = self.compose_at(path + (key,), None, None)
            if result is not None:
                stay.children[key] = result
        if not stay.children:
            return None
        return stay

    def _interior_keys(self, path):
        keys = set()
        for phase in (self.p1, self.p2):
            for other in phase.nodes:
                if other != path and is_super_path(path, other):
                    keys.add(other[len(path)])
        return keys

    # -- lists --------------------------------------------------------------

    def _compose_list(self, path):
        r1, r2 = _record(self.p1, path), _record(self.p2, path)
        adds1 = r1.adds if r1 else {}
        removes1 = r1.removes if r1 else {}
        adds2 = r2.adds if r2 else {}
        removes2 = r2.removes if r2 else {}

        if adds1 and removes1:
            raise UncomposablePatchError(
                'list node {0} has insertions and deletions in the first '
                'patch; no single diff produces that'.format(
                    format_path(path)))
        if adds2 and removes2:
            raise UncomposablePatchError(
                'list node {0} has insertions and deletions in the second '
                'patch'.format(format_path(path)))

        c1 = self._interior_indices(path, self.p1)
        c2 = self._interior_indices(path, self.p2)

        # Phase-1 geometry. A diff only modifies a contiguous trailing region
        # and leaves every lower index in place.
        ins1 = sorted(adds1)
        del1 = sorted(removes1)
        _check_tail_range(ins1, 'insertions', path)
        _check_tail_range(del1, 'deletions', path)
        ins2 = sorted(adds2)
        del2 = sorted(removes2)
        _check_tail_range(ins2, 'insertions', path)
        _check_tail_range(del2, 'deletions', path)
        if adds1:
            n_a = ins1[0]
            n_b = ins1[-1] + 1
        elif removes1:
            n_a = del1[-1] + 1
            n_b = del1[0]
        else:
            # p1 does not resize the list; its length is only visible
            # through phase-2 tail operations or interior change indices.
            if adds2:
                n_b = ins2[0]
            elif removes2:
                n_b = del2[-1] + 1
            else:
                # Include deeper descendants whose first component is a list
                # index (e.g. a change several levels below index 1).
                deep1 = {other[len(path)] for other in self.p1.nodes
                         if other != path and is_super_path(path, other)
                         and len(other) > len(path)}
                deep2 = {other[len(path)] for other in self.p2.nodes
                         if other != path and is_super_path(path, other)
                         and len(other) > len(path)}
                hint_idx = [i for i in c1 | c2 | deep1 | deep2
                            if isinstance(i, int) and not isinstance(i, bool)]
                n_b = max([i + 1 for i in hint_idx] + [1])
            n_a = n_b
        for idx in c1:
            if idx >= n_a:
                raise InvalidPatchError(
                    'interior change beyond list length at {0}'.format(
                        format_path(path + (idx,))))
        for idx in c2:
            if idx >= n_b:
                raise InvalidPatchError(
                    'interior change beyond list length at {0}'.format(
                        format_path(path + (idx,))))

        # Mapping of B indices to their origin after phase 1:
        # ('P', a_idx) for survivors, ('N', value) for phase-1 insertions.
        origin = {}
        for i in range(n_a - len(del1)):
            origin[i] = ('P', i)
        for j, idx in enumerate(ins1):
            origin[n_a - len(del1) + j] = ('N', adds1[idx])

        # A valid p2 diff inserts one contiguous trailing run. It may start
        # at n_b (a normal append) or at the boundary freed by a p1 tail
        # deletion (which becomes an in-place replacement, handled below).
        if ins2 and not (
                ins2 == list(range(ins2[0], ins2[0] + len(ins2)))
                and ins2[0] <= n_b):
            raise UncomposablePatchError(
                'second patch inserts at {0} into a list of length {1} '
                'after the first patch'.format(ins2[0], n_b))
        for idx in del2:
            if idx >= n_b:
                raise ComposeConflict(
                    'second patch deletes list index {0} absent after the '
                    'first patch (length {1})'.format(idx, n_b), path)

        # Deletion at a B index that phase 2 itself also fills is an in-place
        # replacement of the occupant.
        delete_then_add = set(del2) & set(ins2)
        for idx in c2:
            if idx >= n_b:
                raise ComposeConflict(
                    'second patch changes list index {0} absent after the '
                    'first patch (length {1})'.format(idx, n_b), path)

        stay = _R.stay('list')
        net_removed = {}      # A index -> old value
        replaced = {}         # A index -> (old A value, new C value)

        # Phase-1 tail deletions, unless phase 2 fills the freed boundary
        # index again (in-place replacement at that position in B).
        # p1 deleted a trailing run del1 = [n_b, n_b+1, ..., n_a-1]. When p2
        # fills those positions again, they are in-place replacements up to
        # the smaller count; any remaining p2 insertions extend beyond C's
        # overlap with A and are net appends.
        fills = min(len(del1), len(ins2))
        p2_boundary_fill = {ins2[j]: adds2[ins2[j]] for j in range(fills)}
        for idx in del1:
            if idx in p2_boundary_fill:
                replaced[idx] = (removes1[idx], p2_boundary_fill[idx])
            else:
                net_removed[idx] = removes1[idx]

        # Phase-2 tail deletions that did not coincide with a p2 insertion.
        for idx in del2:
            if idx in delete_then_add:
                occ = origin.get(idx)
                if occ is not None and occ[0] == 'P':
                    cc1 = self._child_change(self.p1, path, occ[1])
                    old = cc1[0] if cc1 is not None else removes2[idx]
                    replaced[occ[1]] = (old, adds2[idx])
                # occupant was a p1 insertion: net append with new value
                continue
            occ = origin.get(idx)
            if occ is None:
                raise ComposeConflict(
                    'second patch deletes list index {0} absent after the '
                    'first patch'.format(idx), path)
            if occ[0] == 'P':
                net_removed[occ[1]] = self._revert_p1(
                    path + (occ[1],), removes2[idx])
            # deleting a p1 insertion cancels that insertion

        # In-place replacements compose like leaf changes.
        for a_idx in sorted(replaced):
            old, new = replaced[a_idx]
            cc1 = self._child_change(self.p1, path, a_idx)
            if cc1 is not None:
                t1 = (CHANGE,) + cc1
            else:
                t1 = (CHANGE, old, old)
            result = self._compose_boundary(
                path + (a_idx,), t1, (CHANGE, old, new))
            if result is not None:
                stay.children[a_idx] = result

        # Surviving A slots that were changed in either phase or edited below.
        surviving_a = [i for i in range(n_a) if i not in net_removed
                       and i not in replaced]
        for a_idx in surviving_a:
            b_idx = a_idx  # p1 only deletes tail elements after this index
            cc1 = self._child_change(self.p1, path, a_idx)
            cc2 = self._child_change(self.p2, path, b_idx)
            deeper1 = self._has_children(self.p1, path + (a_idx,))
            deeper2 = self._has_children(self.p2, path + (b_idx,))
            rec1 = _record(self.p1, path + (a_idx,))
            rec2 = _record(self.p2, path + (b_idx,))
            entries1 = bool(rec1 and (rec1.adds or rec1.removes))
            entries2 = bool(rec2 and (rec2.adds or rec2.removes))
            if cc1 is None and cc2 is None and not deeper1 and not deeper2 \
                    and not entries1 and not entries2:
                continue
            t1 = (CHANGE,) + cc1 if cc1 is not None else None
            t2 = (CHANGE,) + cc2 if cc2 is not None else None
            result = self.compose_at(path + (a_idx,), t1, t2)
            if result is not None:
                stay.children[a_idx] = result

        # Net appended values in C order: surviving p1 insertions, then p2
        # inserts that did not replace a deleted element.
        appended = []
        for j, idx in enumerate(ins1):
            b_idx = n_a - len(del1) + j
            if b_idx in del2 and b_idx not in delete_then_add:
                continue  # p1 insertion deleted by p2: cancels
            value = adds1[idx]
            child = path + (b_idx,)
            own_change = self._child_change(self.p2, path, b_idx)
            if own_change is not None:
                self._expect(child, value, own_change[0], 'changed')
                value = own_change[1]
            inner2 = subtree_tuples(self.p2, child,
                                    include_self_entries=True)
            if inner2:
                value = symbolic_apply(inner2, value, child)
            appended.append(value)
        replacement_fill = set(replaced)
        for idx in ins2:
            if idx in replacement_fill:
                continue  # refill of a deleted position: change, not append
            value = adds2[idx]
            child = path + (idx,)
            inner2 = subtree_tuples(self.p2, child,
                                    include_self_entries=True)
            if inner2:
                value = symbolic_apply(inner2, value, child)
            appended.append(value)

        # Drop net removals that actually target a phase-1 insertion (the
        # later deletion cancels that addition).
        n_inserted_b = {n_a - len(del1) + j for j in range(len(ins1))}
        for a_idx in list(net_removed):
            if a_idx in n_inserted_b:
                del net_removed[a_idx]
        # A indices p2 removed already carry the rebuilt A value (computed in
        # the deletion loop). For slots p1 tail-deleted the recorded value is
        # the B value; rebuild their A value when p1 edited the slot.
        rebuilt_via_p2 = {
            origin[i][1] for i in del2
            if origin.get(i) is not None and origin[i][0] == 'P'}
        for a_idx in list(net_removed):
            if a_idx in rebuilt_via_p2:
                continue
            if self._child_change(self.p1, path, a_idx) is not None or \
                    self._has_children(self.p1, path + (a_idx,)):
                net_removed[a_idx] = self._revert_p1(
                    path + (a_idx,), net_removed[a_idx])
        stay.appends = [(n_a - len(net_removed) + j, value)
                        for j, value in enumerate(appended)]
        stay.removes = sorted(net_removed.items(), reverse=True)
        if not stay.children and not stay.appends and not stay.removes:
            return None
        return stay

    def _interior_indices(self, path, phase):
        return {other[-1] for other in phase.nodes
                if other != path and is_super_path(path, other)
                and len(other) == len(path) + 1
                and isinstance(other[-1], int)
                and not isinstance(other[-1], bool)}

    @staticmethod
    def _child_change(phase, path, index):
        record = phase.nodes.get(path + (index,))
        if record is not None and record.change is not None:
            return record.change
        return None

    # -- emission -----------------------------------------------------------

    def _emit(self, result, path, out):
        # Collect (node, action, [(key, value), ...]) entries and emit them
        # grouped per node, exactly like diff does, so swap()/revert work.
        adds = {}    # node -> [(key, value)]
        removes = {}
        changes = []  # (node, (old, new))
        self._collect(result, path, adds, removes, changes)

        # diff emits a depth-first walk: deeper nodes are handled before
        # their parent's own boundary operation, and at one node the order is
        # adds -> (child recursion) -> removes. Build one ordered list of
        # (depth, node, action) items and sort deepest first so interior edits
        # are applied before any parent index shifts.
        items = []
        for node, pair in changes:
            items.append((len(node), node, CHANGE, pair))
        for node, entries in adds.items():
            items.append((len(node), node, ADD,
                          sorted(entries, key=lambda item: (
                              not isinstance(item[0], int), item[0]))))
        for node, entries in removes.items():
            items.append((len(node), node, REMOVE,
                          sorted(entries, key=lambda item: item[0],
                                 reverse=True)))
        # At the same depth, follow diff's recursion: adds before removes;
        # changes sit between them. Deeper paths come first regardless.
        rank = {ADD: 0, CHANGE: 1, REMOVE: 2}
        items.sort(key=lambda item: (-item[0], rank[item[2]]))
        for _, node, action, payload in items:
            out.append((action, _node_name(node), payload))

    def _collect(self, result, path, adds, removes, changes):
        if result.kind == 'dict':
            for key in _sorted_keys(result.children):
                self._collect_child(result.children[key], path, key,
                                    adds, removes, changes)
        elif result.kind == 'list':
            for idx in sorted(result.children):
                self._collect_child(result.children[idx], path, idx,
                                    adds, removes, changes)
            for idx, value in result.appends:
                adds.setdefault(path, []).append((idx, value))
            for idx, old in result.removes:
                removes.setdefault(path, []).append((idx, old))
        elif result.kind == 'set':
            if result.set_add:
                adds.setdefault(path, []).append((0, set(result.set_add)))
            if result.set_remove:
                removes.setdefault(path, []).append(
                    (0, set(result.set_remove)))

    def _collect_child(self, result, parent, key, adds, removes, changes):
        path = parent + (key,)
        if result.kind == 'add':
            adds.setdefault(parent, []).append((key, result.value))
        elif result.kind == 'remove':
            removes.setdefault(parent, []).append((key, result.old))
        elif result.kind == 'change':
            changes.append((path, (result.old, result.new)))
        else:
            self._collect(result, path, adds, removes, changes)


def _check_tail_range(keys, label, path):
    if keys and keys != list(range(keys[0], keys[0] + len(keys))):
        raise UncomposablePatchError(
            'list {0} at {1} are not a contiguous trailing range'.format(
                label, format_path(path)))


# -- Public API -------------------------------------------------------------

def compose(patch1, patch2):
    """Compose two sequential patches ``A -> B`` and ``B -> C``.

    :param patch1: iterable of :func:`dictdiffer.diff` tuples taking *A* to *B*.
    :param patch2: iterable of :func:`dictdiffer.diff` tuples taking *B* to *C*.
    :returns: a list of diff tuples taking *A* directly to *C*. The list is
        minimal: additions later deleted and modifications later reverted are
        cancelled. Old values on surviving ``change``/``remove`` items are the
        original *A* values, so the result is revertible with
        :func:`dictdiffer.revert`.
    :raises ComposeConflict: the second patch's preconditions contradict the
        state produced by the first one.
    :raises UncomposablePatchError: the patches rely on a documented
        unsupported feature (see :mod:`dictdiffer.compose`).
    :raises InvalidPatchError: a patch tuple is malformed.

    >>> from dictdiffer import diff, patch, compose
    >>> a, b, c = {'x': 1}, {'x': 2}, {}
    >>> net = compose(diff(a, b), diff(b, c))
    >>> patch(net, a) == c
    True
    >>> net
    [('remove', '', [('x', 1)])]
    """
    phase1 = _Phase(list(patch1))
    phase2 = _Phase(list(patch2))
    return Composer(phase1, phase2).compose()


def _node_name(path):
    if not path:
        return ''
    if all(isinstance(part, str) and '.' not in part for part in path):
        return '.'.join(path)
    return list(path)


def _sorted_keys(keys):
    return sorted(keys, key=lambda k: (not isinstance(k, str), str(k)))
