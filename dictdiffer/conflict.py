# SPDX-FileCopyrightText: 2015 CERN.
# SPDX-License-Identifier: MIT

"""Sub module to recognize conflicts in dictdiffer patches."""

import itertools

from .numeric import coerce_numeric_policy, take_policy_alias
from .utils import EPSILON, get_path, is_super_path


def _patch_parent_path(patch):
    """Return the structured parent path of a patch."""
    node = patch[1]
    if isinstance(node, str):
        return tuple(node.split('.')) if node else ()
    return tuple(node)


def patches_equivalent(patch1, patch2, numeric_policy):
    """Return ``True`` when both patches make the same numeric decision."""
    if numeric_policy is None or patch1[0] != patch2[0]:
        return False
    if get_path(patch1) != get_path(patch2):
        return False

    action = patch1[0]
    if action == 'change':
        changes1, changes2 = patch1[2], patch2[2]
        if not isinstance(changes1, (tuple, list)) or \
                not isinstance(changes2, (tuple, list)) or \
                len(changes1) != 2 or len(changes2) != 2:
            return False
        return numeric_policy.equal(
            get_path(patch1), changes1[1], changes2[1]
        )

    if action in ('add', 'remove'):
        changes1, changes2 = patch1[2], patch2[2]
        if not isinstance(changes1, (tuple, list)) or \
                not isinstance(changes2, (tuple, list)) or \
                len(changes1) != len(changes2):
            return False
        parent = _patch_parent_path(patch1)
        for change1, change2 in zip(changes1, changes2):
            try:
                key1, value1 = change1
                key2, value2 = change2
            except (TypeError, ValueError):
                return False
            if key1 != key2 or not numeric_policy.equal(
                    parent + (key1,), value1, value2):
                return False
        return True

    return False


class Conflict(object):
    """Wrapper class to store and handle two conflicting patches."""

    def __init__(self, patch1, patch2):
        """Initialize Conflict object.

        :param patch1: First patch tuple
        :param patch2: Second patch tuple
        """
        self.first_patch = patch1
        self.second_patch = patch2
        self.take = None

    def take_patch(self):
        """Return the patch determined by the *take* attribute."""
        if self.take:
            return self.first_patch if self.take == 'f' else self.second_patch
        raise Exception('Take attribute not set.')

    def __repr__(self):
        """Return string representation."""
        return 'Conflict({0}, {1})'.format(self.first_patch, self.second_patch)


class ConflictFinder(object):
    """Responsible for finding conflicting patches."""

    def __init__(self, numeric_policy=None, tolerance=None,
                 absolute_tolerance=None, **policy_aliases):
        """Initialize the conflict finder.

        A numeric policy can be supplied to treat numerically equivalent
        amendments as the same decision.  When no policy or tolerance is
        supplied, conflict detection remains purely path based.
        """
        numeric_policy = take_policy_alias(numeric_policy, policy_aliases)

        if numeric_policy is None and tolerance is None and \
                absolute_tolerance is None:
            self.numeric_policy = None
        else:
            self.numeric_policy = coerce_numeric_policy(
                numeric_policy,
                tolerance=EPSILON if tolerance is None else tolerance,
                absolute_tolerance=absolute_tolerance,
            )

    def _patches_equivalent(self, patch1, patch2):
        """Return ``True`` if both patches make the same numeric decision."""
        return patches_equivalent(
            patch1, patch2, self.numeric_policy
        )

    def _is_conflict(self, patch1, patch2):
        """Decide on a conflict between two patches.

        The conditions are:
        1. The paths are identical
        2. On of the paths is the super path of the other

        :param patch1: First patch tuple
        :param patch2: First patch tuple
        """
        path1 = get_path(patch1)
        path2 = get_path(patch2)

        if path1 == path2:
            return not self._patches_equivalent(patch1, patch2)
        elif is_super_path(path1, path2) and patch1[0] == 'remove':
            return True
        elif is_super_path(path2, path1) and patch2[0] == 'remove':
            return True

        return False

    def find_conflicts(self, first_patches, second_patches):
        """Find all conflicts between two lists of patches.

        Iterates over the lists of patches, comparing each patch from list
        one to each patch from list two.

        :param first_patches: List of patch tuples
        :param second_patches: List of patch tuples
        """
        self.conflicts = [Conflict(patch1, patch2) for patch1, patch2
                          in itertools.product(first_patches,
                                               second_patches)
                          if self._is_conflict(patch1, patch2)]

        return self.conflicts
