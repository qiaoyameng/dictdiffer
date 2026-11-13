# SPDX-License-Identifier: MIT

"""Migrate accumulated patches after renaming nested fields.

A source model may rename nested fields while stored
 :func:`dictdiffer.diff` results still point at the old field names. This
module rewrites the *paths* of such patches according to an explicit mapping
of old paths to new paths. Only paths are touched: the old and new values of
``change`` patches are copied verbatim, including strings that happen to be
named like a renamed field.

Path segments keep their type: string keys stay strings and integer list
indices stay integers. Every mapping is interpreted against the *original*
path simultaneously, which allows exchanging two names (swaps) as well as
moving whole subtrees to new parents. When several mapping entries match a
path, the most specific (longest) source prefix wins.

Diagnostics are returned before any rewritten patch is produced by raising
:class:`MigrationError` with one of the codes:

* ``invalid_mapping`` -- a path is not a tuple of strings and integers;
* ``target_collision`` -- two distinct old paths map to the same new path;
* ``mapping_cycle`` -- the mappings form a cycle other than a pure swap;
* ``lost_context`` -- a patch sits above a mapped field, or an add/remove
  payload needs to descend into a mapped field, so the operation cannot be
  rewritten without inventing context that the patch does not contain.
"""

from collections.abc import MutableMapping, MutableSequence


class MigrationError(ValueError):
    """Raised when patches cannot be unambiguously migrated.

    :ivar code: one of ``invalid_mapping``, ``target_collision``,
                ``mapping_cycle`` or ``lost_context``.
    :ivar path: normalized path involved in the diagnostic, if available.
    :ivar paths: pair of conflicting normalized paths, if available.
    """

    def __init__(self, code, message, path=None, paths=None):
        super(MigrationError, self).__init__(message)
        self.code = code
        self.path = path
        self.paths = paths


def _as_tuple(path, context):
    """Normalize a mapping entry into a typed path tuple."""
    if isinstance(path, str):
        normalized = (path,)
    elif isinstance(path, (list, tuple)):
        normalized = tuple(path)
    else:
        raise MigrationError(
            'invalid_mapping',
            '{!r} in {} is not a sequence of path segments'.format(
                path, context),
            path=tuple(path) if isinstance(path, (list, tuple)) else None)
    for segment in normalized:
        # bool is a subclass of int but never a valid dictdiffer index.
        if isinstance(segment, bool) or not isinstance(segment, (str, int)):
            raise MigrationError(
                'invalid_mapping',
                'path segments must be strings or integers, got {!r} '
                'in {}'.format(segment, context),
                path=normalized)
    return normalized


def _node_tuple(node):
    """Return the typed tuple representation of a patch node."""
    if node == '' or node is None:
        return ()
    if isinstance(node, str):
        return tuple(node.split('.'))
    return tuple(node)


def _has_prefix(path, prefix):
    """Check whether ``prefix`` is a source prefix of ``path``."""
    return len(prefix) <= len(path) and path[:len(prefix)] == prefix


def _longest_prefix(path, sources):
    """Return the longest mapping source that prefixes ``path``."""
    matches = [source for source in sources if _has_prefix(path, source)]
    if not matches:
        return None
    return max(matches, key=len)


def _check_cycles(mapping):
    """Reject cycles, except for pure two-entry swaps.

    Edges follow the longest source prefix of every target, mirroring how
    real paths are interpreted. Identity mappings and pure swaps are safe
    under simultaneous interpretation; longer cycles leave ambiguous where a
    path should end.
    """
    graph = {}
    for source, target in mapping.items():
        prefix = _longest_prefix(target, mapping.keys())
        if prefix is not None:
            edge_target = mapping[prefix] + target[len(prefix):]
            if edge_target != source:
                graph[source] = (prefix, edge_target)

    state = {}
    for start in graph:
        if state.get(start) == 'done':
            continue
        stack = []
        current = start
        while current in graph:
            if state.get(current) == 'done':
                break
            if current in stack:
                cycle = stack[stack.index(current):]
                pure_swap = (
                    len(cycle) == 2
                    and graph[cycle[0]][1] == cycle[1]
                    and graph[cycle[1]][1] == cycle[0])
                if not pure_swap:
                    raise MigrationError(
                        'mapping_cycle',
                        'mapping entries form an ambiguous cycle: '
                        '{}'.format(' -> '.join(map(repr, cycle))),
                        paths=tuple(cycle))
                break
            stack.append(current)
            state[current] = 'visiting'
            current = graph[current][0]
        for node in stack:
            state[node] = 'done'


def _prepare_mapping(mapping):
    """Validate the explicit path mapping once, before touching patches."""
    if isinstance(mapping, MutableMapping):
        items = list(mapping.items())
    else:
        try:
            items = list(mapping)
        except TypeError:
            raise MigrationError(
                'invalid_mapping',
                'mapping must be a dict or an iterable of (old, new) pairs')

    normalized = []
    for old, new in items:
        old_path = _as_tuple(old, 'an old mapping path')
        new_path = _as_tuple(new, 'the new path for {}'.format(old_path))
        normalized.append((old_path, new_path))

    prepared = {}
    for old_path, new_path in normalized:
        if old_path in prepared:
            raise MigrationError(
                'invalid_mapping',
                'duplicate mapping for old path {}'.format(old_path),
                path=old_path)
        prepared[old_path] = new_path

    for index, (old_path, new_path) in enumerate(normalized):
        for other_old, other_new in normalized[index + 1:]:
            if new_path == other_new:
                raise MigrationError(
                    'target_collision',
                    'old paths {} and {} both map to {}'.format(
                        old_path, other_old, new_path),
                    paths=(old_path, other_old))

    _check_cycles(prepared)
    return prepared


def _rewrite_path(path, mapping):
    """Rewrite one full original path using the longest matching prefix."""
    prefix = _longest_prefix(path, mapping.keys())
    if prefix is None:
        return tuple(path)
    return mapping[prefix] + path[len(prefix):]


def _check_add_remove_context(node, payload, mapping):
    """Ensure add/remove payloads explicitly contain every mapped segment."""

    def walk(path, value):
        for source in mapping:
            if not _has_prefix(source, path) or len(source) <= len(path):
                continue
            next_segment = source[len(path)]
            if not isinstance(value, MutableMapping) or \
                    next_segment not in value:
                raise MigrationError(
                    'lost_context',
                    'patch at {} cannot reach mapped field {}; the '
                    'operation would have to descend inside its '
                    'payload'.format(path, source),
                    path=path)
            walk(path + (next_segment,), value[next_segment])

    for key, value in payload:
        entry_path = node + (key,)
        walk(entry_path, value)


def migrate_patch(patch, mapping):
    """Migrate a single dictdiffer patch. See :func:`migrate_patches`."""
    return migrate_patches([patch], mapping)[0]


def migrate_patches(patches, mapping):
    """Rewrite patch paths according to an explicit old/new path mapping.

    :param patches: iterable of ``(action, node, payload)`` tuples as
                    produced by :func:`dictdiffer.diff`.
    :param mapping: dict or iterable of ``(old_path, new_path)`` pairs. Paths
                    are tuples (or lists) whose segments are string keys or
                    integer indices. Mapping entries are interpreted against
                    the original path; the most specific matching prefix
                    wins, so renaming a parent and one of its children works.
    :returns: list of migrated patches. ``add``/``remove`` payloads are
              regrouped by their migrated parent node and relative keys are
              rewritten; ``change`` payloads are returned unchanged.

    >>> from dictdiffer.migration import migrate_patches
    >>> mapping = {('profile', 'name'): ('identity', 'display_name')}
    >>> migrate_patches(
    ...     [('change', ['profile', 'name'], ('old', 'new'))], mapping)
    [('change', ['identity', 'display_name'], ('old', 'new'))]

    Renames are simultaneous, so two names can be exchanged:

    >>> mapping = {('a',): ('b',), ('b',): ('a',)}
    >>> migrate_patches([('change', ['a', 'x'], (1, 2)),
    ...                  ('change', ['b', 'x'], (3, 4))], mapping)
    [('change', ['b', 'x'], (1, 2)), ('change', ['a', 'x'], (3, 4))]

    Relative keys of add/remove payloads follow the same protocol:

    >>> migrate_patches([('add', '', [('a', 1)])],
    ...                 {('a',): ('b',)})
    [('add', '', [('b', 1)])]
    """
    prepared = _prepare_mapping(mapping)
    groups = []
    group_index = {}
    changed_paths = set()

    for action, raw_node, payload in patches:
        node = _node_tuple(raw_node)

        if action == 'change':
            for source in prepared:
                if _has_prefix(source, node) and len(source) > len(node):
                    raise MigrationError(
                        'lost_context',
                        'change at {} cannot reach mapped field {}; the '
                        'mapped field is hidden inside its old/new '
                        'values'.format(node, source),
                        path=node)
        else:
            _check_add_remove_context(node, payload, prepared)

        if action == 'change':
            new_path = _rewrite_path(node, prepared)
            if new_path in changed_paths:
                raise MigrationError(
                    'target_collision',
                    'another patch already targets {}'.format(new_path),
                    path=new_path)
            changed_paths.add(new_path)
            groups.append((action, new_path, payload))
            continue

        for key, value in payload:
            full_path = node + (key,)
            new_path = _rewrite_path(full_path, prepared)
            new_node = new_path[:-1]
            new_key = new_path[-1]
            if (action, new_node) not in group_index:
                group_index[(action, new_node)] = len(groups)
                groups.append((action, new_node, []))
            groups[group_index[(action, new_node)]][2].append(
                (new_key, value))

    migrated = []
    for action, node, payload in groups:
        output_node = list(node) if node else ''
        migrated.append((action, output_node, payload))
    return migrated


def _set_path(container, path, value):
    """Place ``value`` at ``path`` inside a freshly built container."""
    target = container
    for depth, segment in enumerate(path):
        last = depth == len(path) - 1
        if isinstance(segment, int) and not isinstance(segment, bool):
            if target is None:
                target = []
            while len(target) <= segment:
                target.append(None)
            if last:
                target[segment] = value
            elif target[segment] is None:
                target[segment] = [] if isinstance(
                    path[depth + 1], int) and not isinstance(
                        path[depth + 1], bool) else {}
            target = target[segment]
        else:
            if target is None:
                target = {}
            if last:
                target[segment] = value
            elif segment not in target:
                target[segment] = [] if isinstance(
                    path[depth + 1], int) and not isinstance(
                        path[depth + 1], bool) else {}
            target = target[segment]


def _collect_leaves(document, prefix=()):
    """Yield ``(path, value)`` pairs for all leaves of a document."""
    if isinstance(document, MutableMapping):
        if not document:
            yield prefix, document
        for key, value in document.items():
            for item in _collect_leaves(value, prefix + (key,)):
                yield item
    elif isinstance(document, MutableSequence):
        if not document:
            yield prefix, document
        for index, value in enumerate(document):
            for item in _collect_leaves(value, prefix + (index,)):
                yield item
    else:
        yield prefix, document


def migrate_document(document, mapping):
    """Migrate a whole document along the same explicit path mapping.

    This is a structural helper used to verify that migrating a document and
    migrating its patches stay equivalent. It never performs textual JSON
    replacement: values are reinserted at the paths computed from the typed
    segment mapping, and values themselves are left untouched.

    >>> from dictdiffer.migration import migrate_document
    >>> migrate_document({'a': {'x': 1}}, {('a',): ('b',)})
    {'b': {'x': 1}}
    """
    prepared = _prepare_mapping(mapping)
    root = {}
    for path, value in _collect_leaves(document):
        new_path = _rewrite_path(path, prepared)
        if not new_path:
            return value
        _set_path(root, new_path, value)
    return root
