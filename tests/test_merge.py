# SPDX-FileCopyrightText: 2015 CERN.
# SPDX-License-Identifier: MIT

import unittest
from decimal import Decimal

from dictdiffer import NumericTolerance, diff, patch
from dictdiffer.merge import Merger, UnresolvedConflictsException


class MergerTest(unittest.TestCase):
    def test_run(self):
        lca = {'changeme': 'Jo'}
        first = {'changeme': 'Joe'}
        second = {'changeme': 'John'}

        m = Merger(lca, first, second, {})

        self.assertRaises(UnresolvedConflictsException, m.run)

    def test_continue_run(self):
        def take_first(conflict, _, __, ___):
            conflict.take = [('f', x) for x
                             in range(len(conflict.first_patch.patches))]
            return True

        lca = {'changeme': 'Jo'}
        first = {'changeme': 'Joe'}
        second = {'changeme': 'John'}

        m = Merger(lca, first, second, {})

        try:
            m.run()
        except UnresolvedConflictsException:
            pass

        m.continue_run(['f'])

        self.assertEqual(m.unified_patches,
                         [('change', 'changeme', ('Jo', 'Joe'))])

    def test_continue_run_multiple_conflicts_per_patch(self):
        lca = {'foo': [{'x': 1}, {'y': 2}]}
        first = {'foo': [{'x': 1}, {'y': 2}, {'z': 4}]}
        second = {'bar': 'baz'}

        expected = {
            'f': {'foo': [{'x': 1}, {'y': 2}, {'z': 4}],
                  'bar': 'baz'},
            's': {'bar': 'baz'}}

        for resolution, expected_value in expected.items():
            m = Merger(lca, first, second, {})
            try:
                m.run()
            except UnresolvedConflictsException as e:
                m.continue_run([resolution for _ in e.content])

            self.assertEqual(patch(m.unified_patches, lca),
                             expected_value)

    def test_run_with_ignore(self):
        lca = {'changeme': 'Jo', 'ignore': 'Something'}
        first = {'changeme': 'Joe', 'ignore': 'Nothing'}
        second = {'changeme': 'Jo', 'ignore': ''}

        m = Merger(lca, first, second, {}, ignore={'ignore'})

        try:
            m.run()
        except UnresolvedConflictsException:
            self.fail('UnresolvedConflictsException should not be raised')

    def test_tolerance_within_both_sides_is_not_a_conflict(self):
        # Both sides only differ from the lca within the path strategy;
        # the diff drops both changes, hence no conflict and no patch.
        lca = {'amount': Decimal('100.00')}
        first = {'amount': Decimal('100.04')}
        second = {'amount': Decimal('99.97')}

        m = Merger(lca, first, second, {}, tolerances={
            ('amount',): NumericTolerance(abs_tol=Decimal('0.05')),
        })
        m.run()

        self.assertEqual(m.first_patches, [])
        self.assertEqual(m.second_patches, [])
        self.assertEqual(m.conflicts, [])
        self.assertEqual(m.unified_patches, [])

    def test_tolerance_outside_strategy_is_a_conflict(self):
        # One side is beyond the strategy, so exactly one change exists
        # on each side... only the side with a real change - use two
        # conflicting values both outside the strategy.
        lca = {'amount': Decimal('100.00')}
        first = {'amount': Decimal('100.50')}
        second = {'amount': Decimal('99.00')}

        m = Merger(lca, first, second, {}, tolerances={
            ('amount',): NumericTolerance(abs_tol=Decimal('0.05')),
        })
        self.assertRaises(UnresolvedConflictsException, m.run)

        m.continue_run(['f'])
        self.assertEqual(m.unified_patches,
                         [('change', 'amount',
                           (Decimal('100.00'), Decimal('100.50')))])

    def test_tolerance_ignored_change_on_one_side(self):
        # First side is within tolerance (no patch), second side changed;
        # the merge must apply the second side without a conflict.
        lca = {'sensor': {'value': 20.000}}
        first = {'sensor': {'value': 20.0004}}
        second = {'sensor': {'value': 20.5}}

        m = Merger(lca, first, second, {}, tolerances={
            ('sensor', 'value'): NumericTolerance(abs_tol=0.001),
        })
        m.run()

        self.assertEqual(m.first_patches, [])
        self.assertEqual(m.second_patches,
                         [('change', 'sensor.value', (20.000, 20.5))])
        self.assertEqual(m.conflicts, [])
        self.assertEqual(patch(m.unified_patches, lca),
                         {'sensor': {'value': 20.5}})

    def test_diff_and_merge_decisions_agree(self):
        # The same strategy drives both decisions: values the diff
        # considers unchanged cannot produce merge conflicts.
        strategies = {
            ('amount',): NumericTolerance(abs_tol=Decimal('0.05')),
            ('sensor', '*'): NumericTolerance(rel_tol=0.01),
        }
        lca = {'amount': Decimal('100.00'),
               'sensor': {'temp': 20.0}}
        first = {'amount': Decimal('100.03'),
                 'sensor': {'temp': 20.1}}
        second = {'amount': Decimal('100.04'),
                  'sensor': {'temp': 20.1}}

        self.assertEqual(list(diff(lca, first, tolerances=strategies)), [])
        self.assertEqual(list(diff(lca, second, tolerances=strategies)), [])

        m = Merger(lca, first, second, {}, tolerances=strategies)
        m.run()
        self.assertEqual(m.first_patches, [])
        self.assertEqual(m.second_patches, [])
        self.assertEqual(m.conflicts, [])
        self.assertEqual(m.unified_patches, [])

        # Outside the tolerance, both sides change the same path and the
        # merge must surface the conflict the diff also reports.
        first = {'amount': Decimal('100.20'),
                 'sensor': {'temp': 20.1}}
        second = {'amount': Decimal('100.50'),
                  'sensor': {'temp': 20.1}}
        for branch in (first, second):
            self.assertEqual(
                list(diff(lca, branch, tolerances=strategies)),
                [('change', 'amount',
                  (Decimal('100.00'), branch['amount']))])
        m = Merger(lca, first, second, {}, tolerances=strategies)
        self.assertRaises(UnresolvedConflictsException, m.run)

    def test_merge_patch_keeps_target_values_unrounded(self):
        lca = {'amount': Decimal('10.00')}
        first = {'amount': Decimal('10.06')}
        second = {'amount': Decimal('10.00')}

        m = Merger(lca, first, second, {}, tolerances={
            ('amount',): NumericTolerance(abs_tol=Decimal('0.05')),
        })
        m.run()

        self.assertEqual(m.first_patches,
                         [('change', 'amount',
                           (Decimal('10.00'), Decimal('10.06')))])
        self.assertEqual(patch(m.unified_patches, lca),
                         {'amount': Decimal('10.06')})


if __name__ == '__main__':
    unittest.main()
