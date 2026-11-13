"""Path based numeric comparison policies.

This module contains the public primitives used by :func:`dictdiffer.diff`
and ``dictdiffer.merge.Merger`` to compare numeric values.  Policies are
matched against *structured* paths (tuples of dictionary keys and list
indices), so different kinds of data can use different tolerance rules in the
same diff run.

The comparison itself is performed with exact rational arithmetic.  This is
important for ``Decimal`` values: changing the current decimal context must
not change whether two values fall inside a tolerance.
"""

import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from decimal import Decimal
from fractions import Fraction
from typing import Any, Iterable, Optional, Tuple

__all__ = (
    'NumericComparison',
    'NumericPolicy',
    'NumericRule',
    'MatchedNumericRule',
    'IgnoredChange',
    'coerce_numeric_policy',
    'validate_numeric_policy',
    'is_numeric',
)

WILDCARDS = ('*', '+')

POLICY_ALIASES = (
    'numeric_tolerance',
    'numeric_tolerances',
    'numeric_rules',
    'policy',
    'tolerance_policy',
    'tolerance_map',
    'tolerances',
    'comparison_policy',
    'numeric_comparison',
    'path_tolerances',
    'path_rules',
)


def is_numeric(value):
    """Return ``True`` if *value* is a supported numeric value.

    Booleans are deliberately excluded even though ``bool`` is a subclass of
    ``int``.  ``True`` is a boolean value, not the number one.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, Decimal):
        return True
    return isinstance(value, numbers.Real)


def _is_nan_like(value):
    """Return ``True`` for ``NaN`` values, including ``Decimal`` NaN."""
    if isinstance(value, Decimal):
        return value.is_nan()
    if isinstance(value, numbers.Integral) or \
            isinstance(value, numbers.Rational):
        return False
    if isinstance(value, numbers.Real):
        try:
            return math.isnan(value)
        except (TypeError, ValueError, OverflowError):
            return False
    try:
        return bool(value != value)
    except Exception:
        return False


def _is_infinite(value):
    """Return ``True`` for positive or negative infinity."""
    if isinstance(value, Decimal):
        return value.is_infinite()
    if isinstance(value, numbers.Integral) or \
            isinstance(value, numbers.Rational):
        return False
    if isinstance(value, numbers.Real):
        try:
            return math.isinf(value)
        except (TypeError, ValueError, OverflowError):
            return False
    return False


def _is_zero(value):
    """Return ``True`` if a numeric value is zero."""
    if not is_numeric(value):
        return False
    if isinstance(value, Decimal):
        return value.is_zero()
    try:
        return value == 0
    except (TypeError, ValueError):
        return False


def _is_negative(value):
    """Return ``True`` if a numeric value has a negative sign, including -0."""
    if isinstance(value, Decimal):
        return value.is_signed()
    if isinstance(value, numbers.Integral) or \
            isinstance(value, numbers.Rational):
        return value < 0
    if isinstance(value, numbers.Real):
        try:
            return math.copysign(1.0, value) < 0
        except (TypeError, ValueError, OverflowError):
            return False
    return False


def _to_fraction(value):
    """Convert a finite numeric value to an exact ``Fraction``."""
    if isinstance(value, Decimal):
        return Fraction(value)
    if isinstance(value, Fraction):
        return value
    if isinstance(value, numbers.Integral):
        return Fraction(int(value))
    if isinstance(value, float):
        return Fraction.from_float(value)
    if isinstance(value, numbers.Rational):
        return Fraction(value.numerator, value.denominator)
    if isinstance(value, numbers.Real):
        return Fraction.from_float(float(value))
    raise TypeError('unsupported numeric value: {0!r}'.format(value))


def _normalize_tolerance(value, name):
    """Validate and normalize a tolerance value to an exact Fraction."""
    if value is None:
        value = 0
    if isinstance(value, bool):
        raise TypeError('{0} must be a number, not a boolean'.format(name))
    if not is_numeric(value):
        raise TypeError('{0} must be an int, float or Decimal'.format(name))
    if _is_nan_like(value) or _is_infinite(value):
        raise ValueError('{0} must be finite'.format(name))
    fraction = _to_fraction(value)
    if fraction < 0:
        raise ValueError('{0} must not be negative'.format(name))
    return fraction


def _normalize_option(value, allowed, name):
    """Normalize a special-value option name."""
    if isinstance(value, bool):
        if name == 'missing':
            return 'zero' if value else 'diff'
        value = 'equal' if value else 'unequal'
    if not isinstance(value, str):
        raise TypeError('{0} must be a string'.format(name))
    normalized = value.strip().lower().replace('_', '-')
    if name == 'infinity' and normalized in ('same', 'same-sign'):
        normalized = 'same-sign'
    if name == 'infinity' and normalized in ('all', 'any'):
        normalized = 'equal'
    aliases = {
        'same': 'equal',
        'different': 'unequal',
        'distinct': 'distinguish',
    }
    normalized = aliases.get(normalized, normalized)
    if name == 'signed_zero' and normalized == 'unequal':
        normalized = 'distinguish'
    if name == 'missing':
        if normalized in ('add-remove', 'addremove', 'report', 'normal'):
            normalized = 'diff'
        if normalized in ('numeric-zero', 'as-zero', 'treat-as-zero',
                          'ignore'):
            normalized = 'zero'
    if normalized not in allowed:
        raise ValueError(
            '{0} must be one of {1!r}'.format(name, tuple(allowed))
        )
    return normalized


def _reason_code(reason):
    """Map a verbose reason to a short, stable category."""
    if reason.startswith('within-absolute'):
        return 'absolute'
    if reason.startswith('within-relative'):
        return 'relative'
    if reason.startswith('missing-'):
        return 'missing'
    if reason.startswith('nan-'):
        return 'nan'
    if reason.startswith('infinity-'):
        return 'infinity'
    if reason.startswith('signed-zero'):
        return 'signed_zero'
    return reason


@dataclass(frozen=True, init=False)
class NumericRule:
    """Configuration for comparing two numeric values at one path.

    :param absolute_tolerance: Maximum absolute difference considered equal.
    :param relative_tolerance: Relative difference considered equal.  As with
                               ``math.isclose``, the scale is the largest
                               absolute value of the two operands.
    :param nan: ``'equal'`` (default) treats two NaNs as equal; ``'unequal'``
                treats every pair involving NaN as different; ``'ignore'``
                treats any pair involving NaN as equal.
    :param infinity: ``'same-sign'`` (default) treats same-signed infinities
                     as equal; ``'equal'`` treats all infinities as equal;
                     ``'unequal'`` treats all infinities as different;
                     ``'ignore'`` treats any pair involving infinity as equal.
    :param signed_zero: ``'equal'`` (default) treats ``-0`` and ``+0`` as
                        equal; ``'distinguish'`` reports a change.
    :param missing: ``'diff'`` (default) reports missing numeric fields as
                    add/remove operations; ``'zero'`` treats a missing numeric
                    field as zero and suppresses the add/remove when the
                    existing value is within tolerance of zero.
    """

    absolute_tolerance: Any = 0
    relative_tolerance: Any = 0
    nan: str = 'equal'
    infinity: str = 'same-sign'
    signed_zero: str = 'equal'
    missing: str = 'diff'
    _absolute: Fraction = field(default=Fraction(0), repr=False, compare=False)
    _relative: Fraction = field(default=Fraction(0), repr=False, compare=False)

    def __init__(self, absolute_tolerance=0, relative_tolerance=0,
                 nan='equal', infinity='same-sign', signed_zero='equal',
                 missing='diff', **aliases):
        """Initialize and validate a numeric rule."""
        alias_names = {
            'absolute': 'absolute_tolerance',
            'abs': 'absolute_tolerance',
            'abs_tol': 'absolute_tolerance',
            'abs_tolerance': 'absolute_tolerance',
            'absolute_tol': 'absolute_tolerance',
            'relative': 'relative_tolerance',
            'rel': 'relative_tolerance',
            'rel_tol': 'relative_tolerance',
            'rel_tolerance': 'relative_tolerance',
            'relative_tol': 'relative_tolerance',
            'nan_policy': 'nan',
            'nans': 'nan',
            'nan_equal': 'nan',
            'infinity_policy': 'infinity',
            'inf': 'infinity',
            'infinities': 'infinity',
            'infinity_equal': 'infinity',
            'zero_policy': 'signed_zero',
            'zeros': 'signed_zero',
            'zero_equal': 'signed_zero',
            'missing_policy': 'missing',
            'missing_zero': 'missing',
        }
        special = aliases.pop('special', None)
        if special is not None:
            if not isinstance(special, Mapping):
                raise TypeError('special must be a mapping')
            for key, value in special.items():
                canonical = alias_names.get(key, key)
                if canonical == 'absolute_tolerance':
                    absolute_tolerance = value
                elif canonical == 'relative_tolerance':
                    relative_tolerance = value
                elif canonical == 'nan':
                    nan = value
                elif canonical == 'infinity':
                    infinity = value
                elif canonical == 'signed_zero':
                    signed_zero = value
                elif canonical == 'missing':
                    missing = value
                else:
                    raise TypeError(
                        'unexpected special option: {0}'.format(key)
                    )
        for alias, canonical in alias_names.items():
            if alias not in aliases:
                continue
            if canonical != 'absolute_tolerance' and \
                    canonical != 'relative_tolerance':
                current = locals()[canonical]
                if current != NumericRule.__dataclass_fields__[canonical].default:
                    raise TypeError(
                        'both {0} and its alias {1} were given'.format(
                            canonical, alias
                        )
                    )
            value = aliases.pop(alias)
            if canonical == 'absolute_tolerance':
                absolute_tolerance = value
            elif canonical == 'relative_tolerance':
                relative_tolerance = value
            elif canonical == 'nan':
                nan = value
            elif canonical == 'infinity':
                infinity = value
            elif canonical == 'signed_zero':
                signed_zero = value
            elif canonical == 'missing':
                missing = value
        if aliases:
            unexpected = next(iter(aliases))
            raise TypeError('unexpected numeric rule option: {0}'.format(
                unexpected
            ))

        absolute = _normalize_tolerance(
            absolute_tolerance, 'absolute_tolerance'
        )
        relative = _normalize_tolerance(
            relative_tolerance, 'relative_tolerance'
        )
        nan = _normalize_option(
            nan, ('equal', 'unequal', 'ignore'), 'nan'
        )
        infinity = _normalize_option(
            infinity, ('same-sign', 'equal', 'unequal', 'ignore'), 'infinity'
        )
        signed_zero = _normalize_option(
            signed_zero, ('equal', 'distinguish'), 'signed_zero'
        )
        missing = _normalize_option(
            missing, ('diff', 'zero'), 'missing'
        )

        object.__setattr__(self, 'absolute_tolerance', absolute_tolerance)
        object.__setattr__(self, 'relative_tolerance', relative_tolerance)
        object.__setattr__(self, 'nan', nan)
        object.__setattr__(self, 'infinity', infinity)
        object.__setattr__(self, 'signed_zero', signed_zero)
        object.__setattr__(self, 'missing', missing)
        object.__setattr__(self, '_absolute', absolute)
        object.__setattr__(self, '_relative', relative)

    @property
    def absolute(self):
        """Return the absolute tolerance alias."""
        return self.absolute_tolerance

    @property
    def relative(self):
        """Return the relative tolerance alias."""
        return self.relative_tolerance

    @property
    def abs_tol(self):
        """Return the absolute tolerance alias."""
        return self.absolute_tolerance

    @property
    def rel_tol(self):
        """Return the relative tolerance alias."""
        return self.relative_tolerance

    @property
    def special(self):
        """Return the configured special-value options."""
        return {
            'nan': self.nan,
            'infinity': self.infinity,
            'signed_zero': self.signed_zero,
            'missing': self.missing,
        }

    def compare(self, first, second, path=None):
        """Compare two values and explain the decision.

        :return: A :class:`NumericComparison` instance.
        """
        if isinstance(path, str):
            path = tuple(path.split('.')) if path else ()
        else:
            path = tuple(path or ())
        first_numeric = is_numeric(first)
        second_numeric = is_numeric(second)

        first_nan = _is_nan_like(first)
        second_nan = _is_nan_like(second)
        if first_nan or second_nan:
            if self.nan == 'ignore':
                return NumericComparison(
                    True, 'nan-ignored', kind='nan', rule=self, path=path
                )
            if first_nan and second_nan and self.nan == 'equal':
                return NumericComparison(
                    True, 'nan-equal', kind='nan', rule=self, path=path
                )
            return NumericComparison(
                False, 'nan-unequal', kind='nan', rule=self, path=path
            )

        if first_numeric != second_numeric:
            if isinstance(first, bool) or isinstance(second, bool):
                reason = 'boolean-not-numeric'
            else:
                reason = 'non-numeric'
            return NumericComparison(
                False, reason, kind='type-mismatch', rule=self, path=path
            )

        if not first_numeric:
            try:
                equal = bool(first == second)
            except Exception:
                equal = False
            return NumericComparison(
                equal, 'exact' if equal else 'non-numeric',
                kind='non-numeric', rule=self, path=path
            )

        first_infinite = _is_infinite(first)
        second_infinite = _is_infinite(second)
        if first_infinite or second_infinite:
            if self.infinity == 'ignore':
                return NumericComparison(
                    True, 'infinity-ignored', kind='infinity',
                    rule=self, path=path
                )
            if first_infinite and second_infinite:
                if self.infinity == 'equal':
                    return NumericComparison(
                        True, 'infinity-equal', kind='infinity',
                        rule=self, path=path
                    )
                if self.infinity == 'unequal':
                    return NumericComparison(
                        False, 'infinity-unequal', kind='infinity',
                        rule=self, path=path
                    )
                same_sign = (_is_negative(first) == _is_negative(second))
                if same_sign:
                    return NumericComparison(
                        True, 'exact', kind='infinity', rule=self, path=path
                    )
                return NumericComparison(
                    False, 'infinity-sign-mismatch', kind='infinity',
                    rule=self, path=path
                )
            return NumericComparison(
                False, 'infinity-finite', kind='infinity',
                rule=self, path=path
            )

        if _is_zero(first) and _is_zero(second) and \
                _is_negative(first) != _is_negative(second):
            if self.signed_zero == 'distinguish':
                return NumericComparison(
                    False, 'signed-zero-distinct', kind='zero',
                    rule=self, path=path
                )

        try:
            exact_equal = bool(first == second)
        except Exception:
            exact_equal = False
        if exact_equal:
            return NumericComparison(
                True, 'exact', kind='numeric', rule=self, path=path
            )

        first_fraction = _to_fraction(first)
        second_fraction = _to_fraction(second)
        difference = abs(first_fraction - second_fraction)
        scale = max(abs(first_fraction), abs(second_fraction))
        relative_limit = self._relative * scale
        absolute_limit = self._absolute
        allowed = max(absolute_limit, relative_limit)

        if absolute_limit and difference <= absolute_limit:
            return NumericComparison(
                True, 'within-absolute-tolerance', kind='numeric',
                rule=self, path=path, difference=difference,
                allowed=allowed
            )
        if relative_limit and difference <= relative_limit:
            return NumericComparison(
                True, 'within-relative-tolerance', kind='numeric',
                rule=self, path=path, difference=difference,
                allowed=allowed
            )
        if not absolute_limit and not relative_limit:
            reason = 'no-tolerance'
        else:
            reason = 'outside-tolerance'
        return NumericComparison(
            False, reason, kind='numeric', rule=self, path=path,
            difference=difference, allowed=allowed
        )

    def equal(self, first, second, path=None):
        """Return ``True`` when the rule considers the values equal."""
        return self.compare(first, second, path=path).equal

    def is_close(self, first, second, path=None):
        """Return the result of :meth:`equal`."""
        return self.equal(first, second, path=path)

    def to_dict(self):
        """Return a plain dictionary representation of the rule."""
        return {
            'absolute_tolerance': self.absolute_tolerance,
            'relative_tolerance': self.relative_tolerance,
            'nan': self.nan,
            'infinity': self.infinity,
            'signed_zero': self.signed_zero,
            'missing': self.missing,
        }

    def as_dict(self):
        """Return the :meth:`to_dict` result."""
        return self.to_dict()

    @classmethod
    def from_config(cls, config):
        """Build a rule from a mapping, sequence or another rule."""
        return _coerce_rule(config)

    from_dict = from_config


@dataclass(frozen=True)
class NumericComparison:
    """The result of a numeric comparison, including its explanation."""

    equal: bool
    reason: str
    kind: str
    rule: NumericRule
    path: Tuple = field(default_factory=tuple)
    pattern: Optional[Tuple] = None
    difference: Optional[Fraction] = None
    allowed: Optional[Fraction] = None

    def __bool__(self):
        """Allow a comparison to be used directly in boolean context."""
        return self.equal

    @property
    def ignored(self):
        """Return ``True`` if equality was established by a tolerance rule."""
        return self.equal and self.reason != 'exact'

    @property
    def reason_code(self):
        """Return a short category for the comparison reason."""
        return _reason_code(self.reason)

    @property
    def category(self):
        """Return the reason code alias."""
        return self.reason_code

    def to_dict(self):
        """Return a plain dictionary representation of the comparison."""
        return {
            'equal': self.equal,
            'reason': self.reason,
            'kind': self.kind,
            'path': self.path,
            'pattern': self.pattern,
            'difference': self.difference,
            'allowed': self.allowed,
            'rule': self.rule.to_dict(),
        }

    def as_dict(self):
        """Return the :meth:`to_dict` result."""
        return self.to_dict()


@dataclass(frozen=True)
class MatchedNumericRule:
    """A rule selected for a concrete path."""

    rule: NumericRule
    path: Tuple
    pattern: Optional[Tuple] = None
    order: Optional[int] = None

    def to_dict(self):
        """Return a plain dictionary representation of the match."""
        return {
            'rule': self.rule.to_dict(),
            'path': self.path,
            'pattern': self.pattern,
            'order': self.order,
        }

    def as_dict(self):
        """Return the :meth:`to_dict` result."""
        return self.to_dict()


@dataclass(frozen=True)
class IgnoredChange:
    """Description of a change suppressed by a numeric policy."""

    path: Tuple
    first: Any
    second: Any
    reason: str
    rule: NumericRule
    pattern: Optional[Tuple] = None
    difference: Optional[Fraction] = None
    allowed: Optional[Fraction] = None

    @property
    def dotted_path(self):
        """Return a dotted path when all components are plain strings."""
        if all(isinstance(component, str) and '.' not in component
               for component in self.path):
            return '.'.join(self.path)
        return None

    @property
    def path_string(self):
        """Return the dotted path alias."""
        return self.dotted_path

    @property
    def reason_code(self):
        """Return a short category for the ignored-change reason."""
        return _reason_code(self.reason)

    @property
    def category(self):
        """Return the reason code alias."""
        return self.reason_code

    def __getitem__(self, key):
        """Allow dictionary-style access to report fields."""
        return self.to_dict()[key]

    def get(self, key, default=None):
        """Return a report field with a default value."""
        return self.to_dict().get(key, default)

    def to_dict(self):
        """Return a plain dictionary representation of the ignored change."""
        return {
            'path': self.path,
            'dotted_path': self.dotted_path,
            'first': self.first,
            'second': self.second,
            'reason': self.reason,
            'pattern': self.pattern,
            'difference': self.difference,
            'allowed': self.allowed,
            'rule': self.rule.to_dict(),
        }

    def as_dict(self):
        """Return the :meth:`to_dict` result."""
        return self.to_dict()


def _normalize_path(path):
    """Normalize a structured path or dotted string to a tuple."""
    if path is None:
        return ()
    if isinstance(path, str):
        if not path:
            return ()
        return tuple(path.split('.'))
    if isinstance(path, (list, tuple)):
        return tuple(path)
    raise TypeError('path must be a string, list or tuple')


def _normalize_pattern(pattern):
    """Normalize a policy path pattern to a tuple."""
    if pattern is None:
        normalized = ()
    elif isinstance(pattern, (str, list, tuple)):
        normalized = _normalize_path(pattern)
    else:
        normalized = (pattern,)
    for component in normalized:
        if isinstance(component, str) and component not in WILDCARDS and \
                component == '':
            raise ValueError('path patterns must not contain empty parts')
    return normalized


def _path_matches(pattern, path):
    """Return ``True`` when *pattern* matches *path*.

    ``*`` matches one or more path components and ``+`` matches exactly one.
    All other components are compared literally.
    """
    memo = {}

    def match(pattern_index, path_index):
        key = (pattern_index, path_index)
        if key in memo:
            return memo[key]

        if pattern_index == len(pattern):
            result = path_index == len(path)
        else:
            component = pattern[pattern_index]
            if component == '*':
                result = (
                    path_index < len(path) and
                    (match(pattern_index + 1, path_index + 1) or
                     match(pattern_index, path_index + 1))
                )
            elif component == '+':
                result = (
                    path_index < len(path) and
                    match(pattern_index + 1, path_index + 1)
                )
            else:
                result = (
                    path_index < len(path) and
                    component == path[path_index] and
                    match(pattern_index + 1, path_index + 1)
                )
        memo[key] = result
        return result

    return match(0, 0)


def _specificity(pattern, order):
    """Return a deterministic ordering key for a matching pattern."""
    literals = sum(
        1 for component in pattern if component not in WILDCARDS
    )
    pluses = sum(1 for component in pattern if component == '+')
    stars = sum(1 for component in pattern if component == '*')
    return (-literals, -pluses, stars, len(pattern), order)


def _coerce_rule(value):
    """Convert common rule descriptions to a :class:`NumericRule`."""
    if value is None:
        return NumericRule()
    if isinstance(value, NumericRule):
        return value
    if isinstance(value, Mapping):
        return NumericRule(**dict(value))
    if isinstance(value, (tuple, list)):
        if len(value) == 1:
            return NumericRule(relative_tolerance=value[0])
        if len(value) == 2:
            return NumericRule(
                absolute_tolerance=value[0],
                relative_tolerance=value[1],
            )
        raise TypeError('numeric rule sequences need one or two items')
    if isinstance(value, bool):
        raise TypeError('a boolean is not a numeric tolerance')
    if is_numeric(value):
        return NumericRule(relative_tolerance=value)
    raise TypeError('cannot build a numeric rule from {0!r}'.format(value))


def _is_rule_like(value):
    """Return ``True`` if *value* can be coerced to a numeric rule."""
    if isinstance(value, (NumericRule, Mapping)) or value is None:
        return True
    if isinstance(value, bool):
        return False
    if is_numeric(value):
        return True
    if isinstance(value, (tuple, list)) and len(value) in (1, 2):
        return all(is_numeric(item) for item in value)
    return False


def _looks_like_policy_pair(value):
    """Return ``True`` if *value* looks like a ``(pattern, rule)`` pair."""
    return (
        isinstance(value, (tuple, list)) and
        len(value) == 2 and
        _is_rule_like(value[1])
    )


class NumericPolicy:
    """A collection of path patterns and their numeric comparison rules.

    Patterns are structured paths, for example ``('amounts', '*')``, or
    dotted strings such as ``'amounts.*'``.  The wildcard ``*`` matches one
    or more components and ``+`` matches exactly one component.

    When more than one pattern matches a path, the following priority is used
    (all comparisons are deterministic):

    1. more literal components;
    2. more ``+`` components;
    3. fewer ``*`` components;
    4. longer patterns;
    5. earlier registration order.

    An exact path therefore always wins.  Validation only inspects the
    policy description; it never walks the objects that will later be
    diffed.
    """

    def __init__(self, rules=None, default=None, fallback=None, strict=False):
        """Build and validate a policy from rules or (pattern, rule) pairs."""
        if fallback is not None:
            if default is not None:
                raise TypeError('use either default or fallback, not both')
            default = fallback
        self.strict = bool(strict)
        self._entries = []
        self._patterns = set()

        if isinstance(rules, NumericRule):
            if default is not None:
                raise TypeError('default was supplied twice')
            default = rules
            rules = None

        if isinstance(rules, NumericPolicy):
            self._entries = list(rules._entries)
            self._patterns = set(rules._patterns)
            default = rules.default if default is None else default
        elif rules is not None:
            if isinstance(rules, Mapping):
                items = rules.items()
            elif isinstance(rules, tuple) and len(rules) == 2 and \
                    not _looks_like_policy_pair(rules[0]) and \
                    _is_rule_like(rules[1]):
                items = [rules]
            else:
                items = rules
            for order, item in enumerate(items):
                try:
                    if isinstance(item, (tuple, list)) and len(item) == 3:
                        pattern, absolute_tol, relative_tol = item
                        rule = NumericRule(
                            absolute_tolerance=absolute_tol,
                            relative_tolerance=relative_tol,
                        )
                    else:
                        pattern, rule = item
                except (TypeError, ValueError):
                    raise TypeError(
                        'numeric policy entries must be (pattern, rule) pairs'
                    )
                normalized_pattern = _normalize_pattern(pattern)
                if normalized_pattern in self._patterns:
                    raise ValueError(
                        'duplicate numeric policy pattern: {0!r}'.format(
                            normalized_pattern
                        )
                    )
                self._patterns.add(normalized_pattern)
                self._entries.append(
                    (normalized_pattern, _coerce_rule(rule), order)
                )

        if default is None:
            default = NumericRule()
        self.default = _coerce_rule(default)

    @classmethod
    def from_mapping(cls, mapping, default=None):
        """Build a policy from a mapping of patterns to rules."""
        return cls(mapping, default=default)

    @classmethod
    def from_rules(cls, rules, default=None):
        """Build a policy from an iterable of pattern/rule pairs."""
        return cls(rules, default=default)

    @property
    def rules(self):
        """Return the registered ``(pattern, rule)`` pairs."""
        return tuple((pattern, rule)
                     for pattern, rule, _ in self._entries)

    @property
    def default_rule(self):
        """Return the default rule alias."""
        return self.default

    @property
    def patterns(self):
        """Return the registered path patterns."""
        return tuple(pattern for pattern, _, _ in self._entries)

    @property
    def mapping(self):
        """Return registered patterns and rules as a dictionary."""
        return dict(self.rules)

    def __len__(self):
        """Return the number of registered patterns."""
        return len(self._entries)

    def __iter__(self):
        """Iterate over registered ``(pattern, rule)`` pairs."""
        for pattern, rule, _ in self._entries:
            yield pattern, rule

    def __getitem__(self, path):
        """Return the rule selected for *path*."""
        return self.match(path)

    def __contains__(self, path):
        """Return ``True`` when a registered pattern matches *path*."""
        try:
            return self.resolve(path).pattern is not None
        except KeyError:
            return False

    def __eq__(self, other):
        """Compare registered patterns, rules and the default rule."""
        if not isinstance(other, NumericPolicy):
            return NotImplemented
        return (
            self._entries == other._entries and
            self.default == other.default and
            self.strict == other.strict
        )

    def get(self, path, default=None):
        """Return the selected rule or *default* for a strict miss."""
        try:
            return self.match(path)
        except KeyError:
            return default

    def validate(self, *args, **kwargs):
        """Validate the policy without inspecting any business object.

        Any positional or keyword arguments are deliberately ignored: policy
        validation is a description-only operation.
        """
        for pattern, rule, _ in self._entries:
            _normalize_pattern(pattern)
            if not isinstance(rule, NumericRule):
                raise TypeError('numeric policy rules must be NumericRule')
        if not isinstance(self.default, NumericRule):
            raise TypeError('the default numeric rule must be NumericRule')
        return self

    def resolve(self, path):
        """Return the most specific rule for *path*."""
        path = _normalize_path(path)
        matches = [
            (pattern, rule, order)
            for pattern, rule, order in self._entries
            if _path_matches(pattern, path)
        ]
        if not matches:
            if self.strict:
                raise KeyError(
                    'no numeric policy matched path {0!r}'.format(path)
                )
            return MatchedNumericRule(self.default, path)

        pattern, rule, order = min(
            matches, key=lambda entry: _specificity(entry[0], entry[2])
        )
        return MatchedNumericRule(
            rule, path, pattern=pattern, order=order
        )

    def match(self, path):
        """Return just the rule selected for *path*."""
        return self.resolve(path).rule

    def compare(self, path, first, second):
        """Compare two values using the rule selected for *path*."""
        matched = self.resolve(path)
        comparison = matched.rule.compare(first, second, path=matched.path)
        return replace(
            comparison,
            path=matched.path,
            pattern=matched.pattern,
        )

    def equal(self, path, first, second):
        """Return ``True`` when the selected rule considers values equal."""
        return self.compare(path, first, second).equal

    def is_close(self, path, first, second):
        """Return the result of :meth:`equal`."""
        return self.equal(path, first, second)

    def to_dict(self):
        """Return a plain dictionary representation of the policy."""
        return {
            'rules': [
                {'pattern': pattern, 'rule': rule.to_dict()}
                for pattern, rule, _ in self._entries
            ],
            'default': self.default.to_dict(),
        }

    def as_dict(self):
        """Return the :meth:`to_dict` result."""
        return self.to_dict()


def coerce_numeric_policy(policy=None, tolerance=None,
                          absolute_tolerance=None):
    """Build a :class:`NumericPolicy` from a user supplied value.

    ``None`` creates a backward-compatible single default rule from the
    legacy *tolerance* values.  A mapping, iterable of pairs, single rule or
    existing policy is accepted without touching any data object.
    """
    if isinstance(policy, NumericPolicy):
        return policy
    if policy is None:
        default = NumericRule(
            relative_tolerance=0 if tolerance is None else tolerance,
            absolute_tolerance=0 if absolute_tolerance is None
            else absolute_tolerance,
        )
        return NumericPolicy(default=default)
    if isinstance(policy, NumericRule):
        return NumericPolicy(default=policy)
    default = NumericRule(
        relative_tolerance=0 if tolerance is None else tolerance,
        absolute_tolerance=0 if absolute_tolerance is None
        else absolute_tolerance,
    )
    if isinstance(policy, Mapping):
        return NumericPolicy(policy, default=default)
    if isinstance(policy, bool):
        raise TypeError('numeric policy must not be a boolean')
    if is_numeric(policy):
        return NumericPolicy(default=NumericRule(relative_tolerance=policy))
    if isinstance(policy, Iterable):
        return NumericPolicy(policy, default=default)
    raise TypeError('cannot build a numeric policy from {0!r}'.format(policy))


def take_policy_alias(policy, aliases):
    """Resolve supported keyword aliases for the numeric policy.

    This helper keeps older or alternative keyword spellings working without
    making the public signatures unwieldy.  It only looks at the supplied
    keyword mapping, never at business data.
    """
    aliases = dict(aliases)
    selected = None
    for name in POLICY_ALIASES:
        if name in aliases:
            if selected is not None or policy is not None:
                raise TypeError(
                    'multiple numeric policy arguments were supplied'
                )
            selected = aliases.pop(name)
    if aliases:
        unexpected = next(iter(aliases))
        raise TypeError('unexpected keyword argument: {0}'.format(unexpected))
    return policy if policy is not None else selected


def validate_numeric_policy(policy, *args, **kwargs):
    """Validate a policy without inspecting any business object."""
    return coerce_numeric_policy(policy).validate(*args, **kwargs)


NumericComparisonPolicy = NumericPolicy
NumericTolerancePolicy = NumericPolicy
PathNumericPolicy = NumericPolicy
PathTolerance = NumericPolicy
TolerancePolicy = NumericPolicy
NumericTolerance = NumericRule
ToleranceRule = NumericRule
ToleranceSpec = NumericRule
