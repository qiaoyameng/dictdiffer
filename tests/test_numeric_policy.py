"""Tests for path-aware numeric comparison policies."""

from decimal import Decimal, localcontext

import pytest

from dictdiffer import (
    DiffResult,
    IgnoredChange,
    NumericComparisonPolicy,
    NumericPolicy,
    NumericRule,
    NumericTolerance,
    TolerancePolicy,
    diff,
    patch,
)
from dictdiffer.conflict import ConflictFinder
from dictdiffer.merge import Merger, UnresolvedConflictsException
from dictdiffer.utils import PathLimit


def test_amount_and_sensor_rules_are_independent():
    policy = NumericPolicy({
        ('amounts', '*'): NumericRule(
            absolute_tolerance=Decimal('0.01')
        ),
        ('sensors', '*'): NumericRule(
            relative_tolerance=Decimal('0.001')
        ),
    })

    first = {
        'amounts': {'total': Decimal('10.000')},
        'sensors': {'temperature': 100.0},
    }
    second = {
        'amounts': {'total': Decimal('10.005')},
        'sensors': {'temperature': 100.050},
    }
    assert list(diff(first, second, numeric_policy=policy)) == []

    second['amounts']['total'] = Decimal('10.020')
    assert list(diff(first, second, numeric_policy=policy)) == [
        ('change', 'amounts.total',
         (Decimal('10.000'), Decimal('10.020'))),
    ]


def test_wildcard_priority_is_deterministic():
    policy = NumericPolicy([
        (('amounts', '*'), NumericRule(absolute_tolerance=10)),
        (('*',), NumericRule(absolute_tolerance=100)),
        (('amounts', '+'), NumericRule(absolute_tolerance=1)),
        (('amounts', 'value'), NumericRule(absolute_tolerance=0.001)),
        (('a', '*', 'c'), NumericRule(absolute_tolerance=2)),
        (('a', 'b', '*'), NumericRule(absolute_tolerance=3)),
    ])

    assert policy.resolve(('amounts', 'value')).pattern == \
        ('amounts', 'value')
    assert policy.resolve(('amounts', 'other')).pattern == \
        ('amounts', '+')
    assert policy.resolve(('other', 'value')).pattern == ('*',)

    # Both patterns have two literals and one star; registration order wins.
    first_match = policy.resolve(('a', 'b', 'c')).pattern
    assert first_match == ('a', '*', 'c')


def test_policy_validation_does_not_walk_business_objects():
    class ExplodingDict(dict):
        def __iter__(self):
            raise AssertionError('business object was traversed')

        def __getitem__(self, key):
            raise AssertionError('business object was traversed')

    policy = NumericPolicy({('a',): {'absolute_tolerance': Decimal('0.1')}})
    policy.validate()

    with pytest.raises(ValueError):
        diff(
            ExplodingDict({'a': Decimal('1')}),
            ExplodingDict({'a': Decimal('2')}),
            numeric_policy={
                ('a',): {'absolute_tolerance': Decimal('-0.1')}
            },
        )


def test_boolean_is_not_the_number_one():
    assert list(diff({'a': True}, {'a': 1})) == [
        ('change', 'a', (True, 1)),
    ]
    assert list(diff({'a': False}, {'a': 0})) == [
        ('change', 'a', (False, 0)),
    ]
    assert list(diff(
        {'a': True},
        {'a': 1},
        numeric_policy=NumericRule(relative_tolerance=1000),
    )) == [('change', 'a', (True, 1))]
    assert list(diff({'a': True}, {'a': True})) == []

    with pytest.raises(TypeError):
        NumericRule(absolute_tolerance=True)


def test_int_float_and_decimal_are_supported():
    assert list(diff({'a': 1}, {'a': 1.0})) == []
    assert list(diff({'a': Decimal('1.00')}, {'a': 1})) == []
    assert list(diff(
        {'a': 100},
        {'a': Decimal('100.5')},
        numeric_policy=NumericRule(absolute_tolerance=Decimal('1')),
    )) == []
    assert list(diff(
        {'a': 100},
        {'a': Decimal('102')},
        numeric_policy=NumericRule(absolute_tolerance=Decimal('1')),
    )) == [('change', 'a', (100, Decimal('102')))]


def test_decimal_low_precision_and_threshold_boundaries():
    rule = NumericRule(absolute_tolerance=Decimal('0.4'))
    with localcontext() as context:
        context.prec = 3
        first = {'a': Decimal('1000')}

        at_boundary = {'a': Decimal('1000.4')}
        assert list(diff(first, at_boundary, numeric_policy=rule)) == []

        above_boundary = {'a': Decimal('1000.5')}
        assert list(diff(first, above_boundary, numeric_policy=rule)) == [
            ('change', 'a',
             (Decimal('1000'), Decimal('1000.5'))),
        ]

        target = {'a': Decimal('1001.23456789')}
        operations = list(diff(first, target, numeric_policy=rule))
        patched = patch(operations, first)
        assert patched == target
        assert isinstance(patched['a'], Decimal)
        assert patched['a'] == Decimal('1001.23456789')


def test_relative_tolerance_boundary():
    rule = NumericRule(relative_tolerance=Decimal('0.001'))
    first = {'a': Decimal('1000')}
    assert list(diff(first, {'a': Decimal('1001')},
                     numeric_policy=rule)) == []
    assert list(diff(first, {'a': Decimal('1001.1')},
                     numeric_policy=rule)) == [
        ('change', 'a', (Decimal('1000'), Decimal('1001.1'))),
    ]


def test_nan_infinity_and_signed_zero_rules():
    nan = float('nan')
    assert list(diff({'a': nan}, {'a': float('nan')})) == []
    assert list(diff({'a': nan}, {'a': 1.0})) == [
        ('change', 'a', (nan, 1.0)),
    ]
    unequal = list(diff(
        {'a': nan},
        {'a': float('nan')},
        numeric_policy=NumericRule(nan='unequal'),
    ))
    assert len(unequal) == 1
    assert unequal[0][0] == 'change'
    assert unequal[0][1] == 'a'
    assert unequal[0][2][0] != unequal[0][2][0]
    assert unequal[0][2][1] != unequal[0][2][1]

    assert list(diff({'a': float('inf')}, {'a': float('inf')})) == []
    assert list(diff({'a': float('inf')}, {'a': float('-inf')})) == [
        ('change', 'a', (float('inf'), float('-inf'))),
    ]
    assert list(diff(
        {'a': Decimal('Infinity')},
        {'a': Decimal('-Infinity')},
        numeric_policy=NumericRule(infinity='equal'),
    )) == []

    assert list(diff({'a': -0.0}, {'a': 0.0})) == []
    assert list(diff(
        {'a': -0.0},
        {'a': 0.0},
        numeric_policy=NumericRule(signed_zero='distinguish'),
    )) == [('change', 'a', (-0.0, 0.0))]
    assert list(diff({'a': Decimal('-0')}, {'a': Decimal('0')})) == []
    assert list(diff(
        {'a': Decimal('-0')},
        {'a': Decimal('0')},
        numeric_policy=NumericRule(signed_zero='distinguish'),
    )) == [('change', 'a', (Decimal('-0'), Decimal('0')))]


def test_missing_numeric_fields_are_explicit():
    first = {'amounts': {}}
    second = {'amounts': {'total': Decimal('0.004')}}

    assert list(diff(first, second)) == [
        ('add', 'amounts', [('total', Decimal('0.004'))]),
    ]

    rule = NumericRule(
        absolute_tolerance=Decimal('0.01'), missing='zero'
    )
    result = diff(
        first, second,
        numeric_policy=NumericPolicy({('amounts', 'total'): rule}),
        report_ignored=True,
    )
    assert list(result) == []
    assert result.ignored[0].reason == 'missing-treated-as-zero'

    outside = {'amounts': {'total': Decimal('0.02')}}
    assert list(diff(
        first, outside,
        numeric_policy=NumericPolicy({('amounts', 'total'): rule}),
    )) == [('add', 'amounts', [('total', Decimal('0.02'))])]


def test_ignored_change_report_and_actual_values():
    policy = NumericPolicy({
        ('a',): NumericRule(absolute_tolerance=Decimal('0.01'))
    })
    first = {'a': Decimal('1.000')}
    second = {'a': Decimal('1.005'), 'b': Decimal('2.000')}
    result = diff(first, second, numeric_policy=policy,
                  report_ignored=True)

    assert isinstance(result, DiffResult)
    assert list(result) == [
        ('add', '', [('b', Decimal('2.000'))]),
    ]
    assert len(result.ignored) == 1
    report = result.ignored[0]
    assert isinstance(report, IgnoredChange)
    assert report.path == ('a',)
    assert report.dotted_path == 'a'
    assert report.reason == 'within-absolute-tolerance'
    assert report.second == Decimal('1.005')
    assert report.pattern == ('a',)
    assert report['reason'] == 'within-absolute-tolerance'

    sink = []
    list(diff(first, second, numeric_policy=policy,
              ignored_changes=sink))
    assert sink == result.ignored

    callback_sink = []
    list(diff(first, second, numeric_policy=policy,
              report_ignored=callback_sink.append))
    assert callback_sink == result.ignored


def test_path_limit_uses_the_same_policy_for_each_limit():
    policy = NumericPolicy({
        ('a',): NumericRule(absolute_tolerance=Decimal('0.1')),
        ('b',): NumericRule(absolute_tolerance=Decimal('0.1')),
    })
    first = {'a': Decimal('1'), 'b': Decimal('2')}
    second = {'a': Decimal('1.05'), 'b': Decimal('2.2')}
    result = list(diff(
        first, second,
        numeric_policy=policy,
        path_limit=PathLimit([('a',), ('b',)]),
    ))
    assert result == [('change', ['b'],
                       (Decimal('2'), Decimal('2.2')))]

    nested_first = {'a': {'x': Decimal('1.00')}}
    nested_second = {'a': {'x': Decimal('1.005')}}
    nested_policy = NumericPolicy({
        ('a', '*'): NumericRule(absolute_tolerance=Decimal('0.01'))
    })
    assert list(diff(
        nested_first, nested_second,
        numeric_policy=nested_policy,
        path_limit=PathLimit([('a',)]),
    )) == []


def test_merger_and_diff_use_the_same_numeric_decision():
    policy = NumericPolicy({
        ('amount',): NumericRule(absolute_tolerance=Decimal('0.01'))
    })
    lca = {'amount': Decimal('10.000')}
    first = {'amount': Decimal('10.004')}
    second = {'amount': Decimal('10.005')}

    merger = Merger(
        lca, first, second, {},
        numeric_policy=policy,
        report_ignored=True,
    )
    merger.run()
    assert merger.conflicts == []
    assert merger.unified_patches == []
    assert len(merger.ignored_changes) == 2

    conflicting = {}
    merger = Merger(lca, first, second, conflicting)
    with pytest.raises(UnresolvedConflictsException):
        merger.run()

    one_sided = {'amount': Decimal('10.020')}
    merger = Merger(lca, first, one_sided, {}, numeric_policy=policy)
    merger.run()
    assert merger.conflicts == []
    assert merger.unified_patches == [
        ('change', 'amount', (Decimal('10.000'), Decimal('10.020'))),
    ]


def test_conflict_finder_uses_numeric_policy():
    policy = NumericPolicy({
        ('amount',): NumericRule(absolute_tolerance=Decimal('0.01'))
    })
    first = ('change', 'amount',
             (Decimal('10.000'), Decimal('10.004')))
    equivalent = ('change', 'amount',
                  (Decimal('10.000'), Decimal('10.005')))
    different = ('change', 'amount',
                 (Decimal('10.000'), Decimal('10.020')))

    finder = ConflictFinder(numeric_policy=policy)
    assert finder.find_conflicts([first], [equivalent]) == []
    assert len(finder.find_conflicts([first], [different])) == 1


def test_merger_keeps_one_equivalent_structural_patch():
    policy = NumericPolicy({
        ('items', '*'): NumericRule(absolute_tolerance=Decimal('0.01'))
    })
    lca = {'items': []}
    first = {'items': [Decimal('1.004')]}
    second = {'items': [Decimal('1.005')]}

    merger = Merger(lca, first, second, {}, numeric_policy=policy)
    merger.run()

    assert merger.conflicts == []
    assert merger.unified_patches == [
        ('add', 'items', [(0, Decimal('1.004'))]),
    ]
    assert patch(merger.unified_patches, lca) == first


def test_policy_aliases_and_alternate_keywords():
    policy = NumericPolicy({
        ('a',): NumericRule(absolute_tolerance=Decimal('0.1'))
    })
    assert NumericComparisonPolicy is NumericPolicy
    assert TolerancePolicy is NumericPolicy
    assert NumericTolerance is NumericRule

    first = {'a': Decimal('1')}
    second = {'a': Decimal('1.05')}
    assert list(diff(first, second, tolerance=policy)) == []
    assert list(diff(first, second, numeric_tolerances=policy)) == []


def test_rule_validation_errors():
    with pytest.raises(ValueError):
        NumericRule(absolute_tolerance=-1)
    with pytest.raises(ValueError):
        NumericRule(relative_tolerance=float('nan'))
    with pytest.raises(ValueError):
        NumericRule(nan='sometimes')
    with pytest.raises(ValueError):
        NumericPolicy([
            (('a',), {}),
            (('a',), {}),
        ])
