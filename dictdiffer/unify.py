# SPDX-FileCopyrightText: 2015 CERN.
# SPDX-License-Identifier: MIT

"""Sub module to handle the unification of patches after the merge."""

from .conflict import patches_equivalent
from .numeric import coerce_numeric_policy, take_policy_alias
from .utils import get_path, nested_hash


class Unifier(object):
    """Class handling the unification process after the merge."""

    def __init__(self, numeric_policy=None, **policy_aliases):
        """Initialize the unifier with an optional numeric policy."""
        self.numeric_policy = take_policy_alias(
            numeric_policy, policy_aliases
        )
        if self.numeric_policy is not None:
            self.numeric_policy = coerce_numeric_policy(self.numeric_policy)

    def unify(self, first_patches, second_patches, conflicts):
        """Unify two lists of patches into one.

        Takes into account their appearance in the given list of conflicts.

        :param first_patches: list of dictdiffer.diff patches
        :param second_patches: list of dictdiffer.diff patches
        :param conflicts: list of Conflict objects
        """
        self.unified_patches = []
        seen_patches = set()
        self._build_index(conflicts)

        sorted_patches = sorted(first_patches + second_patches, key=get_path)

        for patch in sorted_patches:
            conflict = self._index.get(nested_hash(patch))

            # Apply only the patches that were taken as part of conflict
            # resolution.
            if conflict:
                if conflict.take_patch() != patch:
                    continue

            if self.numeric_policy is not None:
                patch_hash = nested_hash(patch)
                if patch_hash in seen_patches:
                    continue
                if any(
                    patches_equivalent(patch, kept, self.numeric_policy)
                    for kept in self.unified_patches
                ):
                    continue
                seen_patches.add(patch_hash)
            self.unified_patches.append(patch)

        return self.unified_patches

    def _build_index(self, conflicts):
        """Create a dictionary attribute mapping patches to conflicts.

        Creates a dictionary attribute mapping the tuplefied version of each
        patch to it's containing Conflict object.
        """
        self._index = {}
        for conflict in conflicts:
            self._index[nested_hash(conflict.first_patch)] = conflict
            self._index[nested_hash(conflict.second_patch)] = conflict
