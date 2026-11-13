# SPDX-FileCopyrightText: 2026 CERN.
# SPDX-License-Identifier: MIT

"""JSON-safe serialization of dictdiffer patches.

The :func:`dictdiffer.diff` family of functions yields plain Python
objects.  Sending such a patch over a JSON message bus loses type
information: ``set`` and ``tuple`` both become arrays, ``Decimal`` and
``bytes`` have no JSON representation at all, and mapping keys such as
``1`` and ``"1"`` collapse into the same JSON object key.

This module provides :func:`encode_patch` and :func:`decode_patch`, a
self-describing envelope format with an explicit version and a closed
value domain:

* JSON scalars (``None``, ``bool``, ``int``, finite ``float``, ``str``)
  are encoded as-is.
* ``decimal.Decimal`` is encoded exactly, via its string representation.
* ``bytes`` are encoded as base64.
* ``list``, ``tuple``, ``set`` and mappings are always wrapped in
  ``{"$type": ...}`` envelopes, so tags can never be confused with
  ordinary business data.  Mappings are encoded as
  ``{"$type": "map", "entries": [[key, value], ...]}``.
* Mapping keys are restricted to JSON scalars, keeping keys such as
  ``1`` and ``"1"`` distinct.
* Path segments keep their JSON type, so integer and string segments
  stay distinct.

Arbitrary classes, reflection and pickle are intentionally not
supported.

Decoding validates the operation structure, rejects duplicate JSON
object keys, and enforces depth and payload budgets *before* returning
a materialized, executable patch.  It never touches the destination
document, so a rejected message cannot leave a half-applied patch
behind.

>>> from decimal import Decimal
>>> from dictdiffer import diff, patch
>>> from dictdiffer.serialize import decode_patch, encode_patch
>>> first = {'a': {1, 2}, 'b': Decimal('1.10')}
>>> second = {'a': {2, 3}, 'b': Decimal('1.11')}
>>> message = encode_patch(diff(first, second))
>>> decoded = decode_patch(message)
>>> patch(decoded, first) == second
True
"""

import base64
import binascii
import json
import math
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation

FORMAT_NAME = 'dictdiffer-patch'
FORMAT_VERSION = 1

TAG = '$type'

DEFAULT_MAX_MESSAGE_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_DEPTH = 100
DEFAULT_MAX_NODES = 100000

_OP_KEYS = frozenset(('op', 'path', 'changes'))


class PatchEncodeError(ValueError):
    """Raised when a diff result cannot be encoded as a JSON patch."""


class PatchDecodeError(ValueError):
    """Raised when a JSON patch message is invalid or unsupported."""


def _canonical_json(node):
    """Return a deterministic JSON rendering used to sort set items."""
    return json.dumps(node, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def _encode_scalar(value, what):
    """Encode a restricted scalar (mapping key or path segment)."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PatchEncodeError(
                'non-finite floats are not supported: {0}'.format(what))
        return value
    raise PatchEncodeError(
        'unsupported {0} type: {1}'.format(what, type(value).__name__))


def _encode_key(key):
    """Encode a mapping key from the restricted key domain."""
    return _encode_scalar(key, 'mapping key')


def _encode_value(value, depth, max_depth):
    """Encode a patch payload value into the tagged JSON domain."""
    if depth > max_depth:
        raise PatchEncodeError(
            'maximum nesting depth exceeded while encoding')
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PatchEncodeError(
                'non-finite float values are not supported')
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise PatchEncodeError(
                'non-finite Decimal values are not supported')
        return {TAG: 'decimal', 'value': str(value)}
    if isinstance(value, (bytes, bytearray)):
        return {TAG: 'bytes',
                'base64': base64.b64encode(bytes(value)).decode('ascii')}
    if isinstance(value, tuple):
        return {TAG: 'tuple',
                'items': [_encode_value(item, depth + 1, max_depth)
                          for item in value]}
    if isinstance(value, list):
        return {TAG: 'list',
                'items': [_encode_value(item, depth + 1, max_depth)
                          for item in value]}
    if isinstance(value, (set, frozenset)):
        items = [_encode_value(item, depth + 1, max_depth) for item in value]
        items.sort(key=_canonical_json)
        return {TAG: 'set', 'items': items}
    if isinstance(value, Mapping):
        return {TAG: 'map',
                'entries': [[_encode_key(key),
                             _encode_value(item, depth + 1, max_depth)]
                            for key, item in value.items()]}
    raise PatchEncodeError(
        'unsupported value type: {0}'.format(type(value).__name__))


def _encode_node(node):
    """Encode a diff node, keeping path segment types intact."""
    if isinstance(node, str):
        return node
    if isinstance(node, (list, tuple)):
        return [_encode_scalar(segment, 'path segment') for segment in node]
    raise PatchEncodeError(
        'unsupported path node type: {0}'.format(type(node).__name__))


def _encode_op(entry, max_depth):
    """Encode a single diff entry as an operation object."""
    try:
        action, node, changes = entry
    except (TypeError, ValueError):
        raise PatchEncodeError(
            'diff entries must be (action, node, changes) triples')
    if action not in ('add', 'remove', 'change'):
        raise PatchEncodeError(
            'unsupported diff action: {0!r}'.format(action))
    encoded = {'op': action, 'path': _encode_node(node)}
    if action in ('add', 'remove'):
        pairs = []
        for pair in changes:
            try:
                key, value = pair
            except (TypeError, ValueError):
                raise PatchEncodeError(
                    'add/remove changes must be (key, value) pairs')
            pairs.append([_encode_key(key),
                          _encode_value(value, 1, max_depth)])
        encoded['changes'] = pairs
    else:
        try:
            old, new = changes
        except (TypeError, ValueError):
            raise PatchEncodeError(
                'change payload must be an (old, new) pair')
        encoded['changes'] = [_encode_value(old, 1, max_depth),
                              _encode_value(new, 1, max_depth)]
    return encoded


def encode_patch(diff_result, max_depth=DEFAULT_MAX_DEPTH):
    """Encode a diff result as a versioned JSON patch message.

    :param diff_result: Changes returned by ``dictdiffer.diff``.
    :param max_depth: Maximum nesting depth accepted in payload values.
    :returns: The patch message as a JSON string.
    """
    envelope = {
        'format': FORMAT_NAME,
        'version': FORMAT_VERSION,
        'ops': [_encode_op(entry, max_depth) for entry in diff_result],
    }
    return json.dumps(envelope, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':'))


def _reject_duplicate_keys(pairs):
    """Build an object from JSON pairs, rejecting duplicate keys."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise PatchDecodeError(
                'duplicate object key: {0!r}'.format(key))
        result[key] = value
    return result


def _reject_constant(value):
    """Reject non-JSON constants such as NaN and Infinity."""
    raise PatchDecodeError(
        'non-finite number is not valid JSON: {0}'.format(value))


class _Budget(object):
    """Track the remaining payload budget while decoding."""

    def __init__(self, max_nodes):
        """Initialize the budget with a maximum number of nodes."""
        self.remaining = max_nodes

    def consume(self, amount=1):
        """Consume part of the budget, raising when it is exceeded."""
        self.remaining -= amount
        if self.remaining < 0:
            raise PatchDecodeError('payload budget exceeded')


def _envelope_field(node, tag, field):
    """Return the single payload field of a strictly shaped envelope."""
    if set(node) != {TAG, field}:
        raise PatchDecodeError(
            'malformed {0!r} envelope'.format(tag))
    return node[field]


def _decode_key(node):
    """Decode a mapping key from the restricted key domain."""
    if node is None or isinstance(node, (str, bool, int, float)):
        return node
    raise PatchDecodeError(
        'unsupported mapping key type: {0}'.format(type(node).__name__))


def _decode_decimal(node, depth, max_depth, budget):
    """Decode a ``decimal`` envelope into an exact Decimal."""
    raw = _envelope_field(node, 'decimal', 'value')
    if not isinstance(raw, str):
        raise PatchDecodeError('decimal payload must be a string')
    try:
        value = Decimal(raw)
    except InvalidOperation:
        raise PatchDecodeError(
            'invalid decimal payload: {0!r}'.format(raw))
    if not value.is_finite():
        raise PatchDecodeError('non-finite decimal values are not supported')
    return value


def _decode_bytes(node, depth, max_depth, budget):
    """Decode a ``bytes`` envelope from base64."""
    raw = _envelope_field(node, 'bytes', 'base64')
    if not isinstance(raw, str):
        raise PatchDecodeError('bytes payload must be a base64 string')
    try:
        return base64.b64decode(raw.encode('ascii'), validate=True)
    except (UnicodeEncodeError, binascii.Error):
        raise PatchDecodeError('invalid base64 payload')


def _decode_items(node, tag, depth, max_depth, budget):
    """Decode the item list shared by the sequence and set envelopes."""
    items = _envelope_field(node, tag, 'items')
    if not isinstance(items, list):
        raise PatchDecodeError('{0} items must be a list'.format(tag))
    return [_decode_value(item, depth + 1, max_depth, budget)
            for item in items]


def _decode_tuple(node, depth, max_depth, budget):
    """Decode a ``tuple`` envelope."""
    return tuple(_decode_items(node, 'tuple', depth, max_depth, budget))


def _decode_list(node, depth, max_depth, budget):
    """Decode a ``list`` envelope."""
    return _decode_items(node, 'list', depth, max_depth, budget)


def _decode_set(node, depth, max_depth, budget):
    """Decode a ``set`` envelope."""
    try:
        return set(_decode_items(node, 'set', depth, max_depth, budget))
    except TypeError:
        raise PatchDecodeError('set items must be hashable')


def _decode_map(node, depth, max_depth, budget):
    """Decode a ``map`` envelope into a dict."""
    entries = _envelope_field(node, 'map', 'entries')
    if not isinstance(entries, list):
        raise PatchDecodeError('map entries must be a list')
    result = {}
    for entry in entries:
        if not isinstance(entry, list) or len(entry) != 2:
            raise PatchDecodeError(
                'map entries must be [key, value] pairs')
        key = _decode_key(entry[0])
        budget.consume()
        if key in result:
            raise PatchDecodeError(
                'duplicate mapping key after decoding: {0!r}'.format(key))
        result[key] = _decode_value(entry[1], depth + 1, max_depth, budget)
    return result


_DECODERS = {
    'decimal': _decode_decimal,
    'bytes': _decode_bytes,
    'tuple': _decode_tuple,
    'list': _decode_list,
    'set': _decode_set,
    'map': _decode_map,
}


def _decode_value(node, depth, max_depth, budget):
    """Decode a tagged payload value, enforcing depth and budget."""
    budget.consume()
    if node is None or isinstance(node, (str, bool, int, float)):
        return node
    if isinstance(node, list):
        raise PatchDecodeError('unexpected JSON array in value position')
    if not isinstance(node, dict):
        raise PatchDecodeError(
            'unsupported JSON value: {0!r}'.format(node))
    if depth >= max_depth:
        raise PatchDecodeError(
            'maximum nesting depth exceeded while decoding')
    tag = node.get(TAG)
    if not isinstance(tag, str):
        raise PatchDecodeError(
            'objects in value position must be tagged envelopes')
    decoder = _DECODERS.get(tag)
    if decoder is None:
        raise PatchDecodeError('unsupported value tag: {0!r}'.format(tag))
    return decoder(node, depth, max_depth, budget)


def _decode_path(node):
    """Decode a path, keeping integer and string segments distinct."""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        for segment in node:
            if not (segment is None or
                    isinstance(segment, (str, bool, int, float))):
                raise PatchDecodeError('path segments must be JSON scalars')
        return list(node)
    raise PatchDecodeError('path must be a string or a list of segments')


def _decode_op(node, max_depth, budget):
    """Validate and decode a single operation object."""
    budget.consume()
    if not isinstance(node, dict) or set(node) != _OP_KEYS:
        raise PatchDecodeError(
            'operations must be objects with op, path and changes')
    action = node['op']
    if action not in ('add', 'remove', 'change'):
        raise PatchDecodeError(
            'unsupported operation: {0!r}'.format(action))
    path = _decode_path(node['path'])
    changes = node['changes']
    if action in ('add', 'remove'):
        if not isinstance(changes, list):
            raise PatchDecodeError('add/remove changes must be a list')
        decoded_changes = []
        for pair in changes:
            if not isinstance(pair, list) or len(pair) != 2:
                raise PatchDecodeError(
                    'add/remove changes must be [key, value] pairs')
            decoded_changes.append(
                (_decode_key(pair[0]),
                 _decode_value(pair[1], 1, max_depth, budget)))
        return action, path, decoded_changes
    if not isinstance(changes, list) or len(changes) != 2:
        raise PatchDecodeError('change payload must be an [old, new] pair')
    return action, path, (_decode_value(changes[0], 1, max_depth, budget),
                          _decode_value(changes[1], 1, max_depth, budget))


def decode_patch(message, max_message_bytes=DEFAULT_MAX_MESSAGE_BYTES,
                 max_depth=DEFAULT_MAX_DEPTH, max_nodes=DEFAULT_MAX_NODES):
    """Decode a JSON patch message into an executable patch.

    The message is fully validated before anything is returned: the
    operation structure, duplicate object keys, nesting depth and the
    payload budget are all checked first.  The destination document is
    never touched here; apply the returned patch with
    :func:`dictdiffer.patch`.

    :param message: The patch message as a JSON string or UTF-8 bytes.
    :param max_message_bytes: Maximum accepted message size.
    :param max_depth: Maximum nesting depth accepted in payload values.
    :param max_nodes: Maximum number of payload nodes (payload budget).
    :returns: A materialized list of ``(action, node, changes)`` triples
              that can be applied, chained and reused.
    """
    if isinstance(message, (bytes, bytearray)):
        if len(message) > max_message_bytes:
            raise PatchDecodeError('patch message exceeds the size budget')
        try:
            text = bytes(message).decode('utf-8')
        except UnicodeDecodeError:
            raise PatchDecodeError('patch message must be valid UTF-8')
    elif isinstance(message, str):
        if len(message) > max_message_bytes:
            raise PatchDecodeError('patch message exceeds the size budget')
        text = message
    else:
        raise PatchDecodeError('patch message must be str or bytes')
    try:
        envelope = json.loads(text,
                              object_pairs_hook=_reject_duplicate_keys,
                              parse_constant=_reject_constant)
    except json.JSONDecodeError as error:
        raise PatchDecodeError(
            'invalid JSON patch message: {0}'.format(error))
    if not isinstance(envelope, dict):
        raise PatchDecodeError('patch message must be a JSON object')
    if set(envelope) != {'format', 'version', 'ops'}:
        raise PatchDecodeError(
            'patch envelope must contain format, version and ops')
    if envelope['format'] != FORMAT_NAME:
        raise PatchDecodeError('not a dictdiffer patch message')
    version = envelope['version']
    if isinstance(version, bool) or not isinstance(version, int):
        raise PatchDecodeError('patch format version must be an integer')
    if version != FORMAT_VERSION:
        raise PatchDecodeError(
            'unsupported patch format version: {0!r}'.format(version))
    ops = envelope['ops']
    if not isinstance(ops, list):
        raise PatchDecodeError('patch ops must be a list')
    budget = _Budget(max_nodes)
    return [_decode_op(op, max_depth, budget) for op in ops]
