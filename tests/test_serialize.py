# SPDX-FileCopyrightText: 2026 CERN.
# SPDX-License-Identifier: MIT

import json
from decimal import Decimal

import pytest

from dictdiffer import diff, patch, revert
from dictdiffer.serialize import (
    FORMAT_VERSION,
    PatchDecodeError,
    PatchEncodeError,
    decode_patch,
    encode_patch,
)


def make_message(ops, version=FORMAT_VERSION):
    """Build a raw JSON patch message for decode tests."""
    return json.dumps({
        'format': 'dictdiffer-patch',
        'version': version,
        'ops': ops,
    })


def roundtrip(first, second):
    """Encode, decode, patch and revert a diff between two documents."""
    message = encode_patch(diff(first, second))
    assert isinstance(message, str)
    json.loads(message)
    decoded = decode_patch(message)
    assert isinstance(decoded, list)
    assert patch(decoded, first) == second
    assert revert(decoded, second) == first
    return decoded


def test_same_value_different_type_keys():
    """Keys that JSON would conflate keep their types."""
    first = {
        2: 'int',
        '2': 'str',
        1.5: 'float',
        '1.5': 'str-float',
        True: 'bool',
        'true': 'str-bool',
        None: 'none',
        'None': 'str-none',
        'nested': {3: {'deep': [1, 2]}, '3': 'other'},
    }
    second = {
        2: 'int-changed',
        '2': 'str',
        1.5: 'float',
        '1.5': 'str-float-changed',
        True: 'bool',
        'true': 'str-bool',
        None: 'none',
        'None': 'str-none',
        'nested': {3: {'deep': [1, 2, 3]}, '3': 'other'},
        'added': 'new',
    }
    decoded = roundtrip(first, second)
    result = patch(decoded, first)
    key_types = sorted((type(key).__name__, str(key)) for key in result)
    expected = sorted((type(key).__name__, str(key)) for key in second)
    assert key_types == expected
    for key in second:
        assert any(type(key) is type(other) for other in result)


def test_decimal_bytes_tuple_and_set():
    """Exact Decimal, bytes, tuple and set payloads survive a roundtrip."""
    first = {
        'price': Decimal('1.10'),
        'blob': b'\x00\x01\x02',
        'pair': (1, 'a'),
        'tags': {'x', 'y'},
        'nested': {'items': [(1, 2)], 'more': {'deep': {1, 2, 3}}},
    }
    second = {
        'price': Decimal('1.1000000000000000000001'),
        'blob': b'\x00\x01\x03',
        'pair': (1, 'b'),
        'tags': {'y', 'z'},
        'nested': {'items': [(1, 2), (3, 4)], 'more': {'deep': {2, 3, 4}}},
    }
    decoded = roundtrip(first, second)
    result = patch(decoded, first)
    assert isinstance(result['price'], Decimal)
    assert str(result['price']) == '1.1000000000000000000001'
    assert result['blob'] == b'\x00\x01\x03'
    assert isinstance(result['blob'], bytes)
    assert result['pair'] == (1, 'b')
    assert isinstance(result['pair'], tuple)
    assert result['tags'] == {'y', 'z'}
    assert isinstance(result['tags'], set)
    assert result['nested']['more']['deep'] == {2, 3, 4}


def test_set_diff_payload():
    """Set additions and removals carry real set payloads."""
    first = {'s': {1, 2, 3}}
    second = {'s': {2, 3, 4}}
    decoded = roundtrip(first, second)
    payloads = [value for _, _, changes in decoded for _, value in changes]
    assert payloads and all(isinstance(value, set) for value in payloads)


def test_path_segment_types():
    """Integer and string path segments stay distinct."""
    first = {'a': [1, 2], 1: 'x'}
    second = {'a': [1, 3], 1: 'y'}
    decoded = roundtrip(first, second)
    nodes = [node for _, node, _ in decoded]
    assert ['a', 1] in nodes
    assert [1] in nodes
    segment = nodes[nodes.index(['a', 1])][1]
    assert isinstance(segment, int) and not isinstance(segment, bool)
    assert isinstance(nodes[nodes.index([1])][0], int)


def test_dotted_path_kept_verbatim():
    """Dotted string nodes are transported unchanged."""
    decoded = roundtrip({'a': {'x': 1}}, {'a': {'x': 2}})
    assert decoded == [('change', 'a.x', (1, 2))]


def test_tag_and_business_data_are_unambiguous():
    """Business data shaped like a tag is still plain data."""
    first = {'$type': 'decimal', 'nested': {'$type': 'map', 'entries': 'x'}}
    second = {'$type': 'set', 'nested': {'$type': 'map', 'entries': 'y'}}
    result = patch(roundtrip(first, second), first)
    assert result == second
    assert result['nested'] == {'$type': 'map', 'entries': 'y'}


def test_chained_patches_do_not_consume_each_other():
    """Concatenated decoded patches apply in order and stay reusable."""
    doc0 = {'a': 1, 'list': [1, 2]}
    doc1 = {'a': 2, 'list': [1, 2, 3], 'new': 'x'}
    doc2 = {'a': 3, 'list': [1, 2, 3, 4], 'new': 'y', 'extra': {1, 2}}
    patch1 = decode_patch(encode_patch(diff(doc0, doc1)))
    patch2 = decode_patch(encode_patch(diff(doc1, doc2)))
    assert patch(patch1 + patch2, doc0) == doc2
    assert patch(patch1, doc0) == doc1
    assert patch(patch2, doc1) == doc2
    assert revert(patch1, doc1) == doc0
    assert revert(patch2, doc2) == doc1


def test_empty_diff():
    """An empty diff encodes to an empty, reusable patch."""
    decoded = decode_patch(encode_patch(diff({'a': 1}, {'a': 1})))
    assert decoded == []
    assert patch(decoded, {'a': 1}) == {'a': 1}


def test_decode_accepts_bytes_message():
    """Messages may arrive as UTF-8 encoded bytes."""
    message = encode_patch(diff({'a': 1}, {'a': 2}))
    assert decode_patch(message.encode('utf-8')) == [('change', 'a', (1, 2))]


def test_unknown_version_rejected():
    """Messages with an unknown format version are clearly rejected."""
    message = make_message([], version=FORMAT_VERSION + 1)
    with pytest.raises(PatchDecodeError, match='version'):
        decode_patch(message)


def test_bad_type_tag_rejected():
    """Unknown tags and malformed envelopes are clearly rejected."""
    change = {'op': 'change', 'path': 'a', 'changes': [1, 2]}
    bad_values = [
        {'$type': 'wat'},
        {'$type': 'decimal'},
        {'$type': 'decimal', 'value': '1.0', 'extra': 1},
        {'$type': 'decimal', 'value': 'not-a-decimal'},
        {'$type': 'bytes', 'base64': '!!!'},
        {'$type': 'set', 'items': [{'$type': 'map', 'entries': []}]},
        {'value': 'missing tag'},
        [1, 2],
    ]
    for bad in bad_values:
        change['changes'] = [bad, 2]
        with pytest.raises(PatchDecodeError):
            decode_patch(make_message([dict(change)]))


def test_duplicate_keys_rejected():
    """Duplicate JSON object keys are rejected anywhere in the message."""
    message = ('{"format": "dictdiffer-patch", "format": "other",'
               ' "version": 1, "ops": []}')
    with pytest.raises(PatchDecodeError, match='duplicate'):
        decode_patch(message)
    message = ('{"format": "dictdiffer-patch", "version": 1, "ops":'
               ' [{"op": "change", "op": "add", "path": "a",'
               ' "changes": [1, 2]}]}')
    with pytest.raises(PatchDecodeError, match='duplicate'):
        decode_patch(message)


def test_depth_budget_enforced():
    """Deeply nested payloads are rejected before decoding finishes."""
    value = 1
    for _ in range(20):
        value = {'$type': 'list', 'items': [value]}
    message = make_message(
        [{'op': 'change', 'path': 'a', 'changes': [value, 1]}])
    with pytest.raises(PatchDecodeError, match='depth'):
        decode_patch(message, max_depth=5)
    deep = 1
    for _ in range(200):
        deep = {'$type': 'list', 'items': [deep]}
    message = make_message(
        [{'op': 'change', 'path': 'a', 'changes': [deep, 1]}])
    with pytest.raises(PatchDecodeError, match='depth'):
        decode_patch(message)


def test_payload_budget_enforced():
    """Oversized messages and payloads are rejected up front."""
    with pytest.raises(PatchDecodeError, match='size'):
        decode_patch(' ' * 100, max_message_bytes=10)
    items = [{'$type': 'list', 'items': list(range(50))}, 1]
    message = make_message(
        [{'op': 'change', 'path': 'a', 'changes': items}])
    with pytest.raises(PatchDecodeError, match='budget'):
        decode_patch(message, max_nodes=10)


def test_decode_failure_delivers_no_partial_patch():
    """A bad late operation rejects the whole message atomically."""
    ops = [
        {'op': 'change', 'path': 'a', 'changes': [1, 2]},
        {'op': 'change', 'path': 'b', 'changes': [{'$type': 'nope'}, 2]},
    ]
    with pytest.raises(PatchDecodeError):
        decode_patch(make_message(ops))


def test_invalid_structure_rejected():
    """Structurally invalid messages and operations are rejected."""
    with pytest.raises(PatchDecodeError):
        decode_patch('[1, 2, 3]')
    with pytest.raises(PatchDecodeError):
        decode_patch('not json')
    with pytest.raises(PatchDecodeError):
        decode_patch(42)
    bad_ops = [
        {'op': 'change', 'path': 'a'},
        {'op': 'change', 'path': 'a', 'changes': [1, 2], 'extra': 1},
        {'op': 'explode', 'path': 'a', 'changes': [1, 2]},
        {'op': 'change', 'path': {'a': 1}, 'changes': [1, 2]},
        {'op': 'change', 'path': 'a', 'changes': [1]},
        {'op': 'add', 'path': 'a', 'changes': [[1]]},
        {'op': 'add', 'path': 'a', 'changes': [[[1, 2], 1]]},
    ]
    for op in bad_ops:
        with pytest.raises(PatchDecodeError):
            decode_patch(make_message([op]))


def test_non_finite_numbers_rejected():
    """NaN and Infinity are outside the JSON value domain."""
    message = ('{"format": "dictdiffer-patch", "version": 1, "ops":'
               ' [{"op": "change", "path": "a", "changes": [NaN, 1]}]}')
    with pytest.raises(PatchDecodeError):
        decode_patch(message)
    with pytest.raises(PatchEncodeError):
        encode_patch([('change', 'a', (float('nan'), 1))])
    with pytest.raises(PatchEncodeError):
        encode_patch([('change', 'a', (float('inf'), 1))])


def test_encode_rejects_unsupported_types():
    """Arbitrary classes and exotic keys are rejected on encode."""
    class Custom(object):
        pass

    with pytest.raises(PatchEncodeError):
        encode_patch([('change', 'a', (Custom(), 1))])
    with pytest.raises(PatchEncodeError):
        encode_patch([('add', '', [((1, 2), 'x')])])
    with pytest.raises(PatchEncodeError):
        encode_patch([('change', 'a', (Decimal('NaN'), 1))])
    with pytest.raises(PatchEncodeError):
        encode_patch([('explode', 'a', (1, 2))])
    with pytest.raises(PatchEncodeError):
        encode_patch([('change', 'a', 'not-a-pair')])
