import itertools
from copy import deepcopy

import pytest

from dictdiffer import diff, patch, revert
from dictdiffer.optimize import byte_cost, diff_optimal
from dictdiffer.utils import PathLimit


def plan_cost(ops, cost=byte_cost):
    """Return the total cost of a patch plan."""
    return sum(cost(action, node, changes) for action, node, changes in ops)


def assert_roundtrip(first, second, ops):
    """Check that the plan patches and reverts correctly."""
    assert patch(ops, first) == second
    assert revert(ops, second) == first


def assert_no_parent_child_conflict(ops):
    """Check that no replaced node is an ancestor of another change."""
    change_paths = [tuple(node) if isinstance(node, list)
                    else tuple(node.split('.')) if node else ()
                    for action, node, _ in ops if action == 'change']
    for path1, path2 in itertools.combinations(change_paths, 2):
        assert not path1[:len(path2)] == path2
        assert not path2[:len(path1)] == path1


def enumerate_plans(first, second, node=()):
    """Enumerate every legal granularity plan for the subtree at node.

    Legal plans combine, at every dict node, either the fine-grained
    recursion into children or a single replacement of the whole
    subtree (everywhere except at the root).
    """
    if isinstance(first, dict) and isinstance(second, dict):
        combined = [[]]
        for key in first:
            if key in second:
                sub_plans = enumerate_plans(first[key], second[key],
                                            node + (key, ))
                combined = [head + tail for head in combined
                            for tail in sub_plans]
        suffix = []
        additions = [(key, second[key]) for key in second
                     if key not in first]
        if additions:
            suffix.append(('add', list(node), additions))
        deletions = [(key, first[key]) for key in first
                     if key not in second]
        if deletions:
            suffix.append(('remove', list(node), deletions))
        plans = [plan + suffix for plan in combined]
        if node:
            plans.append([('change', list(node), (first, second))])
        return plans
    if first != second:
        return [[('change', list(node), (first, second))]]
    return [[]]


SMALL_CASES = [
    ({'a': {'x': 1, 'y': 2}, 'b': 3},
     {'a': {'x': 9, 'y': 8}, 'b': 3}),
    ({'a': {'x': 1}, 'b': {'y': 2}},
     {'a': {'x': 2}, 'b': {'y': 3, 'z': 4}}),
    ({'a': {'b': {'c': 1, 'd': 2}, 'e': 3}},
     {'a': {'b': {'c': 9}, 'f': 4}}),
    ({'a': {'b': {'c': {'d': 1, 'e': 2}}}},
     {'a': {'b': {'c': {'d': 5, 'f': 6}}}}),
    ({'x': {'p': 1, 'q': 2}, 'y': 1},
     {'x': {'q': 2, 'r': 3}, 'z': 2}),
]


@pytest.mark.parametrize('first,second', SMALL_CASES)
def test_exhaustive_optimality(first, second):
    """The chosen plan is optimal among all legal granularity plans."""
    plans = enumerate_plans(first, second)
    for plan in plans:
        assert_roundtrip(first, second, plan)

    result = diff_optimal(first, second, dot_notation=False)
    assert plan_cost(result) == min(plan_cost(plan) for plan in plans)
    assert_no_parent_child_conflict(result)
    assert_roundtrip(first, second, result)


def test_byte_cost_is_deterministic():
    """The default cost is a deterministic non-negative byte count."""
    assert byte_cost('change', 'a.x', (1, 2)) == \
        len('change a.x (1, 2)'.encode('utf-8'))
    assert byte_cost('change', ['a', 'x'], (1, 2)) == \
        byte_cost('change', 'a.x', (1, 2))
    assert byte_cost('add', '', [('b', 2)]) > 0


def test_large_subtree_replacement():
    """Many fields changing together are cheaper as one replacement."""
    first = {'services': {'db': {'k%d' % i: 'old-%d' % i
                                 for i in range(10)}}}
    second = {'services': {'db': {'k%d' % i: 'new-%d' % i
                                  for i in range(10)}}}
    result = diff_optimal(first, second)
    assert result == [('change', 'services.db',
                       (first['services']['db'], second['services']['db']))]
    assert plan_cost(result) < plan_cost(list(diff(first, second)))
    assert_roundtrip(first, second, result)


def test_few_leaf_changes_stay_fine_grained():
    """A few changed leaves are cheaper as fine-grained changes."""
    first = {'services': {'db': {'k%d' % i: 'value-%d' % i
                                 for i in range(10)}}}
    second = deepcopy(first)
    second['services']['db']['k3'] = 'changed'
    result = diff_optimal(first, second)
    assert result == [('change', 'services.db.k3', ('value-3', 'changed'))]
    assert_roundtrip(first, second, result)


def test_mixed_granularity():
    """Replacement and leaf changes can be picked for sibling subtrees."""
    first = {'a': {'b': {'c': {k: 1 for k in 'vwxyz'}}},
             'other': {'leaf': 1}}
    second = {'a': {'b': {'c': {k: 9 for k in 'vwxyz'}}},
              'other': {'leaf': 2}}
    result = diff_optimal(first, second)
    assert result == [
        ('change', 'a.b.c', ({k: 1 for k in 'vwxyz'},
                             {k: 9 for k in 'vwxyz'})),
        ('change', 'other.leaf', (1, 2)),
    ]
    assert_no_parent_child_conflict(result)
    assert_roundtrip(first, second, result)


def test_ignore_is_hard_constraint():
    """Ignored content is never pulled into the patch to save bytes."""
    first = {'a': {'b': {'c': {'d': dict(
        [('k%d' % i, i) for i in range(10)] + [('secret', 'OLD')])}}}}
    second = {'a': {'b': {'c': {'d': dict(
        [('k%d' % i, i + 100) for i in range(10)] + [('secret', 'NEW')])}}}}

    for ignore in ({'a.b.c.d.secret'}, {('a', 'b', 'c', 'd', 'secret')}):
        result = diff_optimal(first, second, ignore=ignore)
        assert len(result) == 10
        assert all(action == 'change' and node.startswith('a.b.c.d.k')
                   for action, node, _ in result)
        assert 'secret' not in repr(result)
        assert 'OLD' not in repr(result)
        assert 'NEW' not in repr(result)
        # Replacing the subtree would have been cheaper, but it is
        # illegal because it would leak the ignored key.
        illegal = byte_cost('change', 'a.b.c.d',
                            (first['a']['b']['c']['d'],
                             second['a']['b']['c']['d']))
        assert plan_cost(result) > illegal
        patched = patch(result, first)
        assert patched['a']['b']['c']['d']['secret'] == 'OLD'


def test_ignore_whole_subtree():
    """Ignoring a whole subtree yields no operations for it."""
    first = {'a': {'x': 1}, 'b': 1}
    second = {'a': {'x': 2}, 'b': 2}
    assert diff_optimal(first, second, ignore={'a'}) == \
        [('change', 'b', (1, 2))]


def test_path_limit_is_hard_constraint():
    """Differences aggregate at the path_limit, never coarser."""
    first = {'a': {'b': {'c': 'x' * 30, 'd': 'y' * 30}}}
    second = {'a': {'b': {'c': 'u' * 30, 'd': 'v' * 30}}}
    for path_limit in (PathLimit([('a', 'b')]), [('a', 'b')]):
        result = diff_optimal(first, second, path_limit=path_limit)
        assert result == [('change', 'a.b',
                           (first['a']['b'], second['a']['b']))]
        assert_roundtrip(first, second, result)


def test_path_limit_addition_matches_diff():
    """Additions under a path_limit behave exactly like diff."""
    first, second = {}, {'a': {'b': 'c'}}
    for path_limit in (PathLimit(), PathLimit([('a', 'b')]),
                       PathLimit([('a',)])):
        assert diff_optimal(first, second, path_limit=path_limit) == \
            list(diff(first, second, path_limit=path_limit))


def test_set_small_change():
    """Small set changes use member add/remove operations."""
    result = diff_optimal({'s': {1, 2, 3}}, {'s': {1, 2, 3, 4}})
    assert result == [('add', 's', [(0, {4})])]


def test_set_large_change_is_replaced():
    """Large set changes are cheaper as a whole-set replacement."""
    first = {'s': {'old-item-%d' % i for i in range(30)}}
    second = {'s': {'new-item-%d' % i for i in range(30)}}
    result = diff_optimal(first, second)
    assert result == [('change', 's', (first['s'], second['s']))]
    assert_roundtrip(first, second, result)


def test_list_index_change():
    """A single changed list element stays a fine-grained change."""
    first = {'l': list(range(20))}
    second = {'l': list(range(20))}
    second['l'][5] = 999
    result = diff_optimal(first, second)
    assert result == [('change', ['l', 5], (5, 999))]
    assert_roundtrip(first, second, result)


def test_list_many_changes_are_replaced():
    """Many changed list elements are cheaper as one replacement."""
    first = {'l': ['value-%d' % i for i in range(30)]}
    second = {'l': ['other-%d' % i for i in range(30)]}
    result = diff_optimal(first, second)
    assert result == [('change', 'l', (first['l'], second['l']))]
    assert_roundtrip(first, second, result)


def test_custom_cost_function():
    """A caller-provided cost function drives the granularity choice."""
    def cheap_replace(action, node, changes):
        if action == 'change' and isinstance(changes[0], dict):
            return 1
        return 100

    first = {'a': {'x': 1}, 'b': {'y': 2}}
    second = {'a': {'x': 9}, 'b': {'y': 8}}
    result = diff_optimal(first, second, cost=cheap_replace)
    # The root itself is never replaced, its children are.
    assert result == [('change', 'a', ({'x': 1}, {'x': 9})),
                      ('change', 'b', ({'y': 2}, {'y': 8}))]
    assert_roundtrip(first, second, result)
    # Deterministic: repeated calls give the same plan.
    assert diff_optimal(first, second, cost=cheap_replace) == result


def test_root_is_never_replaced():
    """The root node always keeps add/remove granularity."""
    first, second = {'a': 1}, {'b': 2}
    expected = [('add', '', [('b', 2)]), ('remove', '', [('a', 1)])]
    assert diff_optimal(first, second) == expected
    assert diff_optimal(first, second, cost=lambda *args: 1) == expected
    assert_roundtrip(first, second, expected)


def test_empty_diff():
    """Equal objects yield an empty plan."""
    assert diff_optimal({'a': {'x': 1}}, {'a': {'x': 1}}) == []


def test_dot_notation_disabled():
    """Nodes are lists when dot notation is disabled."""
    result = diff_optimal({'a': {'x': 1}}, {'a': {'x': 2}},
                          dot_notation=False)
    assert result == [('change', ['a', 'x'], (1, 2))]


def fine_grained_cost(action, node, changes):
    """Cost model that makes subtree replacement prohibitively costly."""
    if action == 'change' and isinstance(changes[0], (dict, list, set)):
        return 10 ** 9
    return byte_cost(action, node, changes)


CONSISTENCY_CASES = [
    ({'a': {'x': 1, 'y': 2}}, {'a': {'x': 2, 'y': 3}}),
    ({'a': [1, 2, 3]}, {'a': [4, 5]}),
    ({'a': 1, 'b': 2}, {'b': 3, 'c': 4}),
    ({'s': {1, 2, 3}}, {'s': {2, 3, 4}}),
    ({'a': {'b': {'c': 1}}, 'd': [1, {'e': 2}]},
     {'a': {'b': {'c': 2}}, 'd': [1, {'e': 3}], 'f': 'new'}),
    ({'a': 1}, {'a': {'b': 2}}),
    ({'a': {'b': 2}}, {'a': 1}),
]


@pytest.mark.parametrize('first,second', CONSISTENCY_CASES)
def test_matches_diff_when_replacement_is_expensive(first, second):
    """With replacement pessimized the result matches classic diff."""
    result = diff_optimal(first, second, cost=fine_grained_cost)
    assert result == list(diff(first, second))
    assert_roundtrip(first, second, result)
