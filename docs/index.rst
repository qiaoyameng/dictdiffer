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
The tolerance parameter only applies for int and float.

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

The global tolerance is often too coarse when different fields need
different rules. Numeric comparison strategies can be bound to structured
paths with the ``tolerances`` parameter. Each strategy configures an
absolute tolerance, a relative tolerance and the NaN handling
independently, and applies to ``int``, ``float`` and
``decimal.Decimal`` values (``bool`` is never treated as a number):

.. code-block:: python

    from dictdiffer import NumericTolerance

    first = {'price': 100.0, 'sensor': {'temp': 20.001}}
    second = {'price': 100.4, 'sensor': {'temp': 20.002}}

    result = diff(first, second, tolerances={
        ('price',): NumericTolerance(rel_tol=0.001),
        ('sensor', '*'): NumericTolerance(abs_tol=0.01),
    })

    assert list(result) == [('change', 'price', (100.0, 100.4))]

Paths support the ``+`` (single level) and ``*`` (everything that
follows) wildcards. Exact paths take precedence over ``+``, and ``+``
over ``*``. Special values follow explicit rules: signed zeros compare
equal, NaN equals NaN unless ``nan_equal=False``, and infinities are only
equal to an infinity of the same sign. Missing fields are always
reported as ``add``/``remove`` regardless of tolerance, and retained
``change`` operations keep the actual target values - tolerance never
rounds the business data.

An optional report of the changes suppressed by tolerance can be
collected in a list passed as ``ignored_report``. Each entry is a
``(node, (old_value, new_value), strategy)`` tuple:

.. code-block:: python

    report = []
    result = diff(first, second,
                  tolerances={('sensor', '*'): NumericTolerance(abs_tol=0.01)},
                  ignored_report=report)

    assert result == [('change', 'price', (100.0, 100.4))]
    assert report[0][0] == 'sensor.temp'
    assert report[0][1] == (20.001, 20.002)

The same strategies can be handed to the ``Merger`` so that a value
considered unchanged by the diff can never raise a merge conflict:

.. code-block:: python

    from dictdiffer.merge import Merger

    lca = {'price': 100.0, 'sensor': {'temp': 20.001}}
    first = {'price': 100.4, 'sensor': {'temp': 20.002}}
    second = {'price': 100.0, 'sensor': {'temp': 20.003}}

    m = Merger(lca, first, second, {}, tolerances={
        ('sensor', '*'): NumericTolerance(abs_tol=0.01),
    })
    m.run()

API
===

.. automodule:: dictdiffer
   :members:

.. include:: ../CHANGES

.. include:: ../CONTRIBUTING.rst

License
=======

.. include:: ../LICENSE

.. include:: ../AUTHORS
