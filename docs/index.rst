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

Aligning lists of records
-------------------------

By default lists are compared positionally, hence one inserted record
shifts the positions of all following records and the diff reports a
change for each of them. If the lists contain records (dictionaries)
with an identity, the ``align`` parameter accepts a
:class:`.RecordAlignment` strategy that matches records by their
identity instead of their position:

.. code-block:: python

    from dictdiffer import RecordAlignment, diff, patch

    first = [
        {"id": 1, "status": "new"},
        {"id": 2, "status": "new"},
    ]
    second = [
        {"id": 1, "status": "new"},
        {"id": 3, "status": "new"},
        {"id": 2, "status": "done"},
    ]

    result = diff(first, second, align=RecordAlignment("id"))

    assert list(result) == [
        ("add", "", [(1, {"id": 3, "status": "new"})]),
        ("change", [2, "status"], ("new", "done")),
    ]

    # the original to new position mapping is exposed separately
    assert result.mapping == {0: 0, 1: 2}

    assert patch(result, first) == second

The identity does not have to be unique. Records sharing the same
identity are paired so that the total field difference cost is minimal;
ties are resolved deterministically and records that cannot be paired
are reported as additions or removals. The cost of pairing two records
can be customized with the ``field_cost`` parameter.

The result can be used with :func:`.patch`, :func:`.swap` and
:func:`.revert` like a regular diff. Note that :func:`.revert`
reconstructs the original list exactly when the diff does not combine
moved, added and removed records with changes, as the patch format is
positional.

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
