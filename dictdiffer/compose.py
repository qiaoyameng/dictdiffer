# SPDX-FileCopyrightText: 2026 dictdiffer contributors.
# SPDX-License-Identifier: MIT

"""Compose two consecutive dictdiffer patches without the base object.

Given a patch ``p`` transforming ``A -> B`` and a patch ``q`` transforming
``B -> C``, :func:`compose` builds a single patch transforming ``A -> C``.

The composition only inspects the two patches; it never materializes a
fictitious base object.  Whenever the information carried by the patches is
insufficient, a :class:`ComposeUnknown` is raised instead of guessing.  When
``q`` contradicts the intermediate value produced by ``p``, a
:class:`ComposeConflict` is raised.  Both situations are reported explicitly so
that callers can distinguish *unknown* (missing pre-conditions that may be
supplied through the ``prior`` mapping) from *conflicting* (contradictory
instructions that no supplied value can reconcile).
"""

from copy import deepcopy

from . import ADD, CHANGE, REMOVE, patch, revert

# Internal container kinds.
DICT = 'dict'
LIST = 'list'
SET = 'set'

_MISSING = object()


class ComposeError(Exception):
    """Base class for errors raised while composing two patches."""


class ComposeConflict(ComposeError):
    """Raised when the second patch contradicts the intermediate value.

    The first patch already established a value at some path and the second
    patch assumes a different pre-condition at the same path.  Supplying extra
    values cannot fix this; the patches themselves are inconsistent.
    """

    def __init__(self, message, first_patch=None, second_patch=None):
        """Store the offending patches for inspection.

        :param message: Human readable explanation.
        :param first_patch: The instruction from the first patch involved.
        :param second_patch: The instruction from the second patch involved.
        """
        super(ComposeConflict, self).__init__(message)
        self.first_patch = first_patch
        self.second_patch = second_patch


class ComposeUnknown(ComposeError):
    """Raised when the patches alone do not carry enough information.

    The composition needs a pre-condition value that neither patch records.
    The required path is available on the exception and the value can be
    supplied through the ``prior`` argument of :func:`compose`.
    """

    def __init__(self, message, path=None):
        """Store the path whose value is missing.

        :param message: Human readable explanation.
        :param path: Tuple path of the missing intermediate value.
        """
        super(ComposeUnknown, self).__init__(message)
        self.path = tuple(path) if path is not None else None


def _node_keys(node):
    """Return the node as a tuple of path components."""
    if node is None or node == '' or node == []:
        return ()
    if isinstance(node, str):
        return tuple(node.split('.'))
    return tuple(node)


def _render_node(path):
    """Render an internal tuple path back into a dictdiffer node."""
    if not path:
        return ''
    if all(isinstance(part, str) for part in path):
        return '.'.join(path)
    return list(path)


def _expand(patch_result):
    """Yield ``(container, action, key, value)`` single-entry instructions.

    A dictdiffer tuple may carry several changes for one container.  They are
    expanded so that every yielded instruction touches a single entry.  A
    ``change`` targets the node itself and is attached to its parent container.
    """
    for action, node, changes in patch_result:
        container = _node_keys(node)
        if action == CHANGE:
            old_value, new_value = changes
            yield container[:-1], CHANGE, container[-1], \
                (deepcopy(old_value), deepcopy(new_value))
        else:
            for key, value in changes:
                yield container, action, key, deepcopy(value)


def _value_kind(value):
    """Return the container kind represented by an inserted value."""
    if isinstance(value, set):
        return SET
    if isinstance(value, list):
        return LIST
    return DICT


class _Slot(object):
    """Symbolic list slot.

    ``origin`` is the index the slot had in ``A``, or ``None`` for slots
    inserted by the patches.  ``original`` records the value in ``A`` whenever
    it becomes known from a ``change`` or ``remove`` instruction, otherwise it
    stays ``None``.  Base slots that are only ever read by position are lazy:
    their ``value`` starts as ``_MISSING`` and is bound on first use.
    """

    def __init__(self, origin, value=_MISSING, original=None):
        self.origin = origin
        self.value = value
        self.original = original
        self.inserted = origin is None


class _Composer(object):
    """Stateful engine composing two expanded instruction streams."""

    def __init__(self, prior, verify):
        self.prior = {tuple(path): deepcopy(value)
                      for path, value in (prior or {}).items()}
        # When verifying, pre-conditions that cannot be derived from the
        # patches raise :class:`ComposeUnknown` instead of being trusted.
        self.verify = verify
        self.first = {}
        self.second = {}
        self.result = []

    def build(self, first_patch, second_patch):
        """Index both streams and compose starting from the root container."""
        for instruction in _expand(first_patch):
            self.first.setdefault(instruction[0], []).append(instruction)
        for instruction in _expand(second_patch):
            self.second.setdefault(instruction[0], []).append(instruction)

        self._compose_container((), self._root_kind())
        return self.result

    def _root_kind(self):
        """Infer the root container kind from the available instructions."""
        for groups in (self.first, self.second):
            for container, instructions in groups.items():
                if container:
                    continue
                for _, action, _, value in instructions:
                    if action == CHANGE:
                        # A root change targets the container itself; the
                        # replacement value reveals the root container kind.
                        return _value_kind(value[1])
                    if action == ADD:
                        return _value_kind(value)
        # A root list whose length is unchanged only carries changes, which
        # are indexed by integer keys under the empty root node.
        for groups in (self.first, self.second):
            for container, instructions in groups.items():
                if container == () and instructions and \
                        isinstance(instructions[0][2], int):
                    return LIST
        # No root-level structural instruction; default to a dictionary, which
        # is the dominant dictdiffer use case.
        return DICT

    def _prior(self, path):
        """Return a deep copy of a supplied prior value, or ``_MISSING``."""
        if path in self.prior:
            return deepcopy(self.prior[path])
        return _MISSING

    def _compose_container(self, path, kind):
        """Compose all instructions touching one container and emit tuples."""
        if kind == LIST:
            self._compose_list(path)
        elif kind == SET:
            self._compose_set(path)
        else:
            self._compose_dict(path)

    # -- dictionaries -----------------------------------------------------

    def _compose_dict(self, path):
        first = {instr[2]: instr for instr in self.first.get(path, [])}
        second = {instr[2]: instr for instr in self.second.get(path, [])}
        keys = list(first)
        for key in second:
            if key not in first:
                keys.append(key)

        adds = []
        for key in keys:
            emitted = self._compose_entry(path, key,
                                          first.get(key), second.get(key))
            if emitted is not None:
                adds.append(emitted)
        if adds:
            self.result.append((ADD, _render_node(path), adds))

    def _compose_entry(self, path, key, p_instr, q_instr):
        """Compose one dictionary entry; return an add pair or emit in place."""
        child = path + (key,)

        if p_instr is None:
            return self._only_one_side(path, key, q_instr, True)
        if q_instr is None:
            return self._only_one_side(path, key, p_instr, False)

        p_action = p_instr[1]
        q_action = q_instr[1]
        p_value = p_instr[3]
        q_value = q_instr[3]

        if p_action == ADD and q_action == ADD:
            raise ComposeConflict(
                'Second patch adds {0}, already added by the first patch.'
                .format(child), p_instr, q_instr)
        if p_action == REMOVE and q_action in (REMOVE, CHANGE):
            raise ComposeConflict(
                'Second patch {0}s {1}, removed by the first patch.'
                .format(q_action, child), p_instr, q_instr)
        if p_action == ADD and q_action == REMOVE:
            if q_value != p_value:
                raise ComposeConflict(
                    'Second patch removes {0!r} but the first patch added '
                    '{1!r}.'.format(q_value, p_value), p_instr, q_instr)
            return None  # addition and removal of the same value cancel
        if p_action == ADD and q_action == CHANGE:
            if q_value[0] != p_value:
                raise ComposeConflict(
                    'Second patch changes newly added {0} from {1!r}, not '
                    '{2!r}.'.format(child, p_value, q_value[0]),
                    p_instr, q_instr)
            return key, deepcopy(q_value[1])
        if p_action == CHANGE and q_action in (CHANGE, REMOVE):
            p_old, p_new = p_value
            q_old = q_value[0] if q_action == CHANGE else q_value
            if q_old != p_new:
                raise ComposeConflict(
                    'Second patch expects {0} to be {1!r} but the first '
                    'patch left {2!r}.'.format(child, q_old, p_new),
                    p_instr, q_instr)
            if q_action == CHANGE:
                self.result.append(
                    (CHANGE, _render_node(child),
                     (deepcopy(p_old), deepcopy(q_value[1]))))
            else:
                self.result.append(
                    (REMOVE, _render_node(path),
                     [(key, deepcopy(p_old))]))
            return None

        # Structural entries whose descendants are also modified.
        return self._compose_nested(path, key, p_instr, q_instr)

    def _only_one_side(self, path, key, instruction, from_second):
        """Emit an instruction that no other patch touches at this entry."""
        _, action, _, value = instruction
        child = path + (key,)
        if from_second and action in (REMOVE, CHANGE) and self.verify:
            expected = value if action == REMOVE else value[0]
            supplied = self._prior(child)
            if supplied is _MISSING or supplied != expected:
                raise ComposeUnknown(
                    'Cannot verify the pre-condition of the second patch at '
                    '{0}; supply it through "prior".'.format(child), child)
        if action == ADD:
            return key, deepcopy(value)
        if action == REMOVE:
            self.result.append(
                (REMOVE, _render_node(path), [(key, deepcopy(value))]))
            return None
        self.result.append(
            (CHANGE, _render_node(child),
             (deepcopy(value[0]), deepcopy(value[1]))))
        return None

    def _compose_nested(self, path, key, p_instr, q_instr):
        """Handle structural entries with descendant instructions."""
        child = path + (key,)
        p_action = p_instr[1]
        q_action = q_instr[1]

        if p_action == REMOVE:
            raise ComposeConflict(
                'Second patch modifies {0}, removed by the first patch.'
                .format(child), p_instr, q_instr)

        if p_action == ADD:
            final_value = patch(self._nested_tuples(self.second, child),
                                deepcopy(p_instr[3]))
            return key, final_value

        if p_action == CHANGE:
            p_old, p_new = p_instr[3]
            q_old = q_instr[3][0] if q_action == CHANGE else q_instr[3]
            if q_action in (CHANGE, REMOVE) and q_old != p_new:
                raise ComposeConflict(
                    'Second patch expects {0} to be {1!r} but the first '
                    'patch left {2!r}.'.format(child, q_old, p_new),
                    p_instr, q_instr)
            if q_action == REMOVE:
                self.result.append(
                    (REMOVE, _render_node(path),
                     [(key, deepcopy(p_old))]))
                return None
            if q_action == ADD:
                raise ComposeConflict(
                    'Second patch adds existing entry {0}.'.format(child),
                    p_instr, q_instr)
            final_value = patch(self._nested_tuples(self.second, child),
                                deepcopy(p_new))
            self.result.append(
                (CHANGE, _render_node(child),
                 (deepcopy(p_old), final_value)))
            return None

        # The first patch only modifies descendants of an existing container.
        if q_action == ADD:
            raise ComposeConflict(
                'Second patch adds existing entry {0}.'.format(child),
                p_instr, q_instr)
        if q_action == REMOVE:
            original = revert(self._nested_tuples(self.first, child),
                              deepcopy(q_instr[3]))
            self.result.append(
                (REMOVE, _render_node(path), [(key, original)]))
            return None
        if q_action == CHANGE:
            q_old, q_new = q_instr[3]
            original = revert(self._nested_tuples(self.first, child),
                              deepcopy(q_old))
            self.result.append(
                (CHANGE, _render_node(child),
                 (original, deepcopy(q_new))))
            return None

        # Neither patch restructures the entry itself; both only touch its
        # descendants, so the entry stays an existing container and we compose
        # one level deeper.
        if child in self.first or child in self.second:
            self._compose_container(child, self._infer_child_kind(child))
        return None

    def _infer_child_kind(self, path):
        """Infer a nested container kind from surrounding instructions."""
        saw_integer_key = False
        for groups in (self.first, self.second):
            for container, instructions in groups.items():
                if container[:len(path)] != path:
                    continue
                for _, action, entry_key, value in instructions:
                    if action == CHANGE:
                        return _value_kind(value[1])
                    if action == ADD and container == path:
                        return _value_kind(value)
                    if container == path and isinstance(entry_key, int):
                        saw_integer_key = True
        return LIST if saw_integer_key else DICT

    @staticmethod
    def _nested_tuples(groups, prefix):
        """Rebuild absolute dictdiffer tuples for descendants of ``prefix``."""
        tuples = []
        for container, instructions in groups.items():
            if container[:len(prefix)] != prefix or container == prefix:
                continue
            grouped = {}
            for _, action, entry_key, value in instructions:
                if action == CHANGE:
                    tuples.append(
                        (CHANGE, _render_node(container[:-1]),
                         (container[-1], value)))
                else:
                    grouped.setdefault(action, []).append(
                        (entry_key, value))
            for action, changes in grouped.items():
                tuples.append(
                    (action, _render_node(container), changes))
        return tuples

    # -- sets -------------------------------------------------------------

    def _compose_set(self, path):
        added = set()
        removed = set()
        for _, action, _, value in self.first.get(path, []):
            if action == ADD:
                added |= value
                removed -= value
            else:
                removed |= value
                added -= value
        for _, action, _, value in self.second.get(path, []):
            if action == ADD:
                if value & removed:
                    raise ComposeConflict(
                        'Second patch re-adds set members removed by the '
                        'first patch at {0}.'.format(path))
                added |= value
            else:
                if value & added:
                    added -= value
                else:
                    removed |= value
        node = _render_node(path)
        if removed:
            self.result.append((REMOVE, node, [(0, removed)]))
        if added:
            self.result.append((ADD, node, [(0, added)]))

    # -- lists ------------------------------------------------------------

    def _compose_list(self, path):
        # Origin-in-A -> original value, for every deleted base slot.
        removed = {}
        replay = {'inserts': 0, 'base_removed': 0}
        slots = self._replay_first(path, removed, replay)
        slots = self._replay_second(path, slots, removed, replay)
        self._emit_list(path, slots, removed)

    def _replay_first(self, path, removed, replay):
        """Replay first-patch instructions into a symbolic slot sequence."""
        slots = []
        for _, action, key, value in self.first.get(path, []):
            if action == ADD:
                slots.insert(key, _Slot(None, value))
                replay['inserts'] += 1
                continue
            slots = self._materialize(slots, key, replay)
            slot = slots[key]
            if action == REMOVE:
                self._expect(slot, value, path, key)
                if not slot.inserted:
                    removed[slot.origin] = value
                    replay['base_removed'] += 1
                del slots[key]
            else:
                old_value, new_value = value
                self._expect(slot, old_value, path, key)
                if not slot.inserted:
                    slot.original = old_value
                slot.value = new_value
        return slots

    def _replay_second(self, path, slots, removed, replay):
        """Replay second-patch instructions over the symbolic slots."""
        for _, action, key, value in self.second.get(path, []):
            if action == ADD:
                slots.insert(key, _Slot(None, value))
                replay['inserts'] += 1
                continue
            slots = self._materialize(slots, key, replay)
            slot = slots[key]
            if action == REMOVE:
                self._verify_slot(slot, value, path, key)
                if not slot.inserted:
                    removed[slot.origin] = value
                    replay['base_removed'] += 1
                del slots[key]
            else:
                old_value, new_value = value
                self._verify_slot(slot, old_value, path, key)
                if not slot.inserted and slot.original is None:
                    slot.original = old_value
                slot.value = new_value
        return slots

    @staticmethod
    def _materialize(slots, key, replay):
        """Ensure a base slot exists at position ``key``.

        Lazily created slots fill gaps left by base entries the patches never
        describe.  The current position is a position in the post-patch
        sequence, so the slot's origin in ``A`` is recovered by undoing the
        net index shift recorded so far::

            origin = position - live_inserts + base_removals
        """
        while len(slots) <= key:
            position = len(slots)
            origin = (position - replay['inserts']
                      + replay['base_removed'])
            slots.append(_Slot(origin))
        return slots

    @staticmethod
    def _expect(slot, expected, path, key):
        """Check a first-patch pre-condition against a modeled slot."""
        if slot.value is not _MISSING and slot.value != expected:
            raise ComposeConflict(
                'First patch expects {0!r} at {1}[{2}] but the slot holds '
                '{3!r}.'.format(expected, path, key, slot.value))
        if slot.value is _MISSING:
            slot.value = expected

    def _verify_slot(self, slot, expected, path, key):
        """Check a second-patch pre-condition against a modeled slot."""
        if slot.inserted:
            if slot.value != expected:
                raise ComposeConflict(
                    'Second patch expects {0!r} at {1}[{2}] but the first '
                    'patch inserted {3!r}.'.format(
                        expected, path, key, slot.value))
            return
        if slot.value is not _MISSING:
            if slot.value != expected:
                raise ComposeConflict(
                    'Second patch expects {0!r} at {1}[{2}] but the first '
                    'patch left {3!r}.'.format(
                        expected, path, key, slot.value))
            return
        if self.verify:
            # The first patch never established this slot, so q's pre-condition
            # cannot be derived from the patches and must be supplied.
            supplied = self._prior(path + (key,))
            if supplied is _MISSING or supplied != expected:
                raise ComposeUnknown(
                    'Cannot verify list pre-condition at {0}[{1}].'
                    .format(path, key), path + (key,))
        # Outside verification, trust the value recorded by the second patch.
        slot.value = expected

    def _emit_list(self, path, slots, removed):
        """Emit change/remove/add tuples reproducing the A -> C transition."""
        node = _render_node(path)
        changes = []
        adds = []
        live_origins = set()

        for position, slot in enumerate(slots):
            if slot.inserted:
                adds.append((position, deepcopy(slot.value)))
                continue
            live_origins.add(slot.origin)
            if slot.original is not None and slot.original != slot.value:
                changes.append(
                    (CHANGE, _render_node(path + (slot.origin,)),
                     (deepcopy(slot.original), deepcopy(slot.value))))

        removes = [(origin, value) for origin, value in removed.items()
                   if origin not in live_origins]

        for change in changes:
            self.result.append(change)
        if removes:
            self.result.append((REMOVE, node, sorted(removes, reverse=True)))
        if adds:
            self.result.append((ADD, node, sorted(adds)))


def compose(first_patch, second_patch, prior=None, verify=False):
    """Compose two consecutive patches into a single equivalent patch.

    ``first_patch`` transforms an object ``A`` into ``B`` and
    ``second_patch`` transforms ``B`` into ``C``.  The returned iterator
    yields standard dictdiffer tuples transforming ``A`` directly into ``C``,
    usable as-is with :func:`dictdiffer.patch` and :func:`dictdiffer.revert`.

    Neither ``A`` nor a complete ``B`` is required.  The only accepted inputs
    are the two patches and, optionally, the handful of intermediate values
    listed in ``prior`` (keyed by tuple path) when the patches alone are
    ambiguous.

    >>> from dictdiffer import compose
    >>> p = [('change', 'a', (1, 2))]
    >>> q = [('change', 'a', (2, 3))]
    >>> list(compose(p, q))
    [('change', 'a', (1, 3))]

    Add followed by remove cancels, remove followed by modification conflicts,
    and contradictory pre-conditions raise :class:`ComposeConflict` rather than
    being guessed.

    :param first_patch: Iterable of dictdiffer tuples (``A -> B``).  Consumed
        once, so one-shot generators are accepted.
    :param second_patch: Iterable of dictdiffer tuples (``B -> C``).  Consumed
        once.
    :param prior: Optional mapping of tuple path to the intermediate (``B``)
        value required at paths the patches cannot describe on their own.
    :param verify: When true, pre-conditions that cannot be derived from the
        patches raise :class:`ComposeUnknown` even when they could otherwise be
        trusted; ``prior`` always resolves them.
    :raises ComposeConflict: The two patches make contradictory assumptions.
    :raises ComposeUnknown: A pre-condition cannot be determined from the
        patches and was not supplied through ``prior``.
    """
    composer = _Composer(prior, verify)
    return iter(composer.build(list(first_patch), list(second_patch)))
