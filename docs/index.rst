==========
Dictdiffer
==========
.. currentmodule:: dictdiffer

.. image:: https://github.com/inveniosoftware/dictdiffer/workflows/CI/badge.svg
        :target: https://github.com/inveniosoftware/dictdiffer/actions

.. image:: https://img.shields.io/coveralls/inveniosoftware/dictdiffer.svg
        :target: https://coveralls.io/r/inveniosoftware/dictdiffer

.. image:: https://img.shields.io/github/tag/inveniosoftware/dictdiffer.svg
        :target: https://github.com/inveniosoftware/dictdiffer/releases

.. image:: https://img.shields.io/pypi/dm/dictdiffer.svg
        :target: https://pypi.python.org/pypi/dictdiffer

.. image:: https://img.shields.io/github/license/inveniosoftware/dictdiffer.svg
        :target: https://github.com/inveniosoftware/dictdiffer/blob/master/LICENSE

Dictdiffer is a helper module that helps you to diff and patch
dictionaries.

Installation
============

Dictdiffer is on PyPI so all you need is:

.. code-block:: console

    $ pip install dictdiffer


Usage
=====

Let's start with an example on how to find the diff between two
dictionaries using :func:`.diff` method:

.. code-block:: python

    from dictdiffer import diff, patch, swap, revert

    first = {
        "title": "hello",
        "fork_count": 20,
        "stargazers": ["/users/20", "/users/30"],
        "settings": {
            "assignees": [100, 101, 201],
        }
    }

    second = {
        "title": "hellooo",
        "fork_count": 20,
        "stargazers": ["/users/20", "/users/30", "/users/40"],
        "settings": {
            "assignees": [100, 101, 202],
        }
    }

    result = diff(first, second)

    assert list(result) == [
        ('change', ['settings', 'assignees', 2], (201, 202)),
        ('add', 'stargazers', [(2, '/users/40')]),
        ('change', 'title', ('hello', 'hellooo'))]


Now we can apply the diff result with :func:`.patch` method:

.. code-block:: python

    result = diff(first, second)
    patched = patch(result, first)

    assert patched == second


Also we can swap the diff result with :func:`.swap` method:

.. code-block:: python

    result = diff(first, second)
    swapped = swap(result)

    assert list(swapped) == [
        ('change', ['settings', 'assignees', 2], (202, 201)),
        ('remove', 'stargazers', [(2, '/users/40')]),
        ('change', 'title', ('hellooo', 'hello'))]


Let's revert the last changes:

.. code-block:: python

    result = diff(first, second)
    reverted = revert(result, patched)
    assert reverted == first

A tolerance can be used to consider closed values as equal.
The tolerance parameter applies to int, float and Decimal values.

Let's try with a tolerance of 10% with the values 10 and 10.5:

.. code-block:: python

    first = {'a': 10.0}
    second = {'a': 10.5}

    result = diff(first, second, tolerance=0.1)

    assert list(result) == []

Now with a tolerance of 1%:

.. code-block:: python

    result = diff(first, second, tolerance=0.01)

    assert list(result) == ('change', 'a', (10.0, 10.5))

Path-aware numeric policies
---------------------------

A single global tolerance is often wrong when one structure contains, for
example, monetary amounts and sensor readings.
:class:`dictdiffer.numeric.NumericPolicy` maps structured paths to independent
:class:`dictdiffer.numeric.NumericRule` objects:

.. code-block:: python

    from decimal import Decimal

    from dictdiffer import NumericPolicy, NumericRule, diff

    policy = NumericPolicy({
        ('amounts', '*'): NumericRule(
            absolute_tolerance=Decimal('0.01'),
            relative_tolerance=Decimal('0'),
        ),
        ('sensors', '*'): NumericRule(
            absolute_tolerance=Decimal('0'),
            relative_tolerance=Decimal('0.001'),
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

Rules support ``int``, ``float`` and ``Decimal``.  Booleans are not numbers:
``True`` is never treated as ``1``.  Comparison is performed with exact
rational arithmetic, so a low-precision ``Decimal`` context cannot change a
tolerance decision or round the values written by :func:`dictdiffer.patch`.

Each rule configures these independent decisions:

``absolute_tolerance``
    The maximum absolute difference considered equal.

``relative_tolerance``
    The maximum difference relative to the largest absolute operand, using the
    same scale as ``math.isclose``.

``nan``
    ``'equal'`` (default) treats two NaNs as equal; ``'unequal'`` treats every
    pair involving a NaN as different; ``'ignore'`` treats any pair involving
    a NaN as equal.

``infinity``
    ``'same-sign'`` (default) treats equally signed infinities as equal;
    ``'equal'`` treats all infinities as equal; ``'unequal'`` treats all
    infinities as different; ``'ignore'`` treats any pair involving infinity
    as equal.

``signed_zero``
    ``'equal'`` (default) treats ``-0`` and ``+0`` as equal;
    ``'distinguish'`` reports a change.

``missing``
    ``'diff'`` (default) reports a missing numeric field as an add/remove
    operation.  ``'zero'`` treats the missing side as zero and suppresses the
    operation when the present value is within tolerance of zero.

Patterns are matched against structured paths, for example
``('amounts', '*')``, or dotted strings such as ``'amounts.*'``.  ``*``
matches one or more path components and ``+`` matches exactly one.  When
several patterns match, precedence is deterministic: more literal
components, then more ``+`` components, then fewer ``*`` components, then a
longer pattern, then registration order.  An exact path always wins.

Policy validation only inspects the policy description; it never walks the
business objects that are later diffed.

When a raw mapping is supplied, unmatched paths fall back to the legacy
``tolerance``/``absolute_tolerance`` values.  A
:class:`dictdiffer.numeric.NumericPolicy` uses its own ``default`` rule
instead.

An optional report lists changes that were ignored by a numeric rule:

.. code-block:: python

    result = diff(
        first, second,
        numeric_policy=policy,
        report_ignored=True,
    )
    assert list(result) == []
    ignored = result.ignored
    assert ignored[0].path == ('amounts', 'total')
    assert ignored[0].reason == 'within-absolute-tolerance'

The report records the structured path, the matched wildcard pattern, the
configured rule, the exact difference and the allowed threshold.  Reports can
also be sent to a list or callback with ``ignored_changes=...``.

The same policy can be supplied to ``dictdiffer.merge.Merger``.  Patch
extraction and conflict detection then use the same numeric decision:

.. code-block:: python

    from dictdiffer.merge import Merger

    merger = Merger(
        lca, first, second, {},
        numeric_policy=policy,
        report_ignored=True,
    )
    merger.run()
    assert merger.conflicts == []
    assert merger.ignored_changes

Changes that are kept still contain the actual target value.  Tolerance only
decides whether a change is reported; it never rounds or rewrites business
data.

API
===

.. automodule:: dictdiffer
   :members:

.. automodule:: dictdiffer.numeric
   :members:

.. include:: ../CHANGES

.. include:: ../CONTRIBUTING.rst

License
=======

.. include:: ../LICENSE

.. include:: ../AUTHORS
