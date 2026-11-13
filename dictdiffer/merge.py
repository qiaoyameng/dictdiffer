# SPDX-FileCopyrightText: 2015 CERN.
# SPDX-FileCopyrightText: 2017 ETH Zurich, Swiss Data Science Center, Jiri Kuncar.
# SPDX-License-Identifier: MIT

"""Sub module to handle the merging of dictdiffer patches."""

from . import diff
from .conflict import ConflictFinder
from .numeric import coerce_numeric_policy, take_policy_alias
from .resolve import Resolver, UnresolvedConflictsException
from .unify import Unifier
from .utils import EPSILON, PathLimit


class Merger(object):
    """Class wrapping steps of the automated merging process.

    Usage:
        >>> lca = {}
        >>> first = {'foo': 'bar'}
        >>> second = {'bar': 'foo'}
        >>> path_limits = []
        >>> actions = {}
        >>> additional_info = {}
        >>> m = Merger(lca, first, second, actions,
        ...            path_limits, additional_info)
        >>> try:
        ...     m.run()
        ... except UnresolvedConflictsException:
        ...     # fix the conflicts
        ...     m.continue_run()
    """

    def __init__(self,
                 lca, first, second, actions,
                 path_limits=[], additional_info=None, ignore=None,
                 numeric_policy=None, tolerance=None,
                 absolute_tolerance=None, report_ignored=None,
                 ignored_changes=None, **policy_aliases):
        """Initialize the Merger object.

        :param lca: latest common ancestor of the two diverging data structures
        :param first: first data structure
        :param second: second data structure
        :param path_limits: list of paths, utilized to instantiate a
                            dictdiffer.utils.PathLimit object
        :param additional_info: Any object containing additional information
                                used by the resolution functions
        :param ignore: Set of keys that should not be merged
        :param numeric_policy: Numeric comparison policy used for both
                               patch extraction and conflict detection.
        :param tolerance: Legacy relative tolerance used when no numeric
                          policy is given.
        :param absolute_tolerance: Legacy absolute tolerance used when no
                                   numeric policy is given.
        :param report_ignored: If true, collect changes ignored because of
                               numeric tolerance in *ignored_changes*.
        :param ignored_changes: Additional list or callback report sink.
        """
        self.lca = lca
        self.first = first
        self.second = second
        self.path_limit = PathLimit(path_limits)
        self.ignore = ignore
        self.report_ignored = report_ignored
        self._ignored_sink = ignored_changes

        numeric_policy = take_policy_alias(numeric_policy, policy_aliases)

        self.numeric_policy = coerce_numeric_policy(
            numeric_policy,
            tolerance=EPSILON if tolerance is None else tolerance,
            absolute_tolerance=absolute_tolerance,
        )

        self.actions = actions
        self.additional_info = additional_info

        self.conflict_finder = ConflictFinder(
            numeric_policy=self.numeric_policy
        )

        self.resolver = Resolver(self.actions,
                                 self.additional_info,
                                 numeric_policy=self.numeric_policy)

        self.unifier = Unifier(numeric_policy=self.numeric_policy)

        self.conflicts = []
        self.unresolved_conflicts = []
        self.ignored_changes = []
        self.first_ignored_changes = []
        self.second_ignored_changes = []

    def run(self):
        """Run the automated merging process.

        Runs every step necessary for the automated merging process, raising
        an UnresolvedConflictsException in case that the provided resolution
        actions can not solve a given conflict.

        After every performed step, the results are stored inside attributes of
        the merger object.
        """
        self.extract_patches()
        self.find_conflicts()
        self.resolve_conflicts()

        if self.unresolved_conflicts:
            raise UnresolvedConflictsException(self.unresolved_conflicts)

        self.unify_patches()

    def continue_run(self, picks):
        """Continue the merge after an UnresolvedConflictsException.

        :param picks: a list of 'f' or 's' strings, which utilize the Conflicts
                      class *take* attribute
        """
        self.resolver.manual_resolve_conflicts(picks)
        self.unresolved_conflicts = []
        self.unify_patches()

    def extract_patches(self):
        """Extract the patches.

        Extracts the differences between the *lca* and the *first* and
        *second* data structure and stores them in the attributes
        *first_patches* and *second_patches*.
        """
        report_options = {}
        report_sinks = []
        if self.report_ignored is True:
            report_options['report_ignored'] = True
        elif self.report_ignored is not None and self.report_ignored is not False:
            report_sinks.append(self.report_ignored)
        if self._ignored_sink is not None:
            report_sinks.append(self._ignored_sink)
        if report_sinks:
            report_options['report_ignored'] = True
        if len(report_sinks) == 1:
            report_options['ignored_changes'] = report_sinks[0]
        elif report_sinks:

            def _report(record):
                for sink in report_sinks:
                    if hasattr(sink, 'append'):
                        sink.append(record)
                    elif hasattr(sink, 'add'):
                        sink.add(record)
                    else:
                        sink(record)

            report_options['ignored_changes'] = _report

        first_result = diff(
            self.lca, self.first,
            path_limit=self.path_limit,
            ignore=self.ignore,
            expand=True,
            numeric_policy=self.numeric_policy,
            **report_options
        )
        self.first_patches = list(first_result)
        self.first_ignored_changes = first_result.ignored or []

        second_result = diff(
            self.lca, self.second,
            path_limit=self.path_limit,
            ignore=self.ignore,
            expand=True,
            numeric_policy=self.numeric_policy,
            **report_options
        )
        self.second_patches = list(second_result)
        self.second_ignored_changes = second_result.ignored or []

        reporting = (
            self.report_ignored is not None and
            self.report_ignored is not False
        ) or self._ignored_sink is not None
        if reporting:
            self.ignored_changes = (
                list(self.first_ignored_changes) +
                list(self.second_ignored_changes)
            )

    def find_conflicts(self):
        """Find conflicts between the tow lists of patches.

        Finds the conflicts between the two difference lists and stores
        them in the *conflicts* attribute.
        """
        self.conflicts = (self
                          .conflict_finder
                          .find_conflicts(self.first_patches,
                                          self.second_patches))

    def resolve_conflicts(self):
        """Resolve the conflicts.

        Runs the automated conflict resolution process.
        Occurring unresolvable conflicts are stored in *unresolved_conflicts*.
        """
        try:
            self.resolver.resolve_conflicts(self.first_patches,
                                            self.second_patches,
                                            self.conflicts)
        except UnresolvedConflictsException as e:
            self.unresolved_conflicts = e.content

    def unify_patches(self):
        """Unify the patches after the conflict resolution.

        Unifies the patches after a successful merge and stores them in
        *unified_patches*.
        """
        self.unified_patches = self.unifier.unify(self.first_patches,
                                                  self.second_patches,
                                                  self.conflicts)
