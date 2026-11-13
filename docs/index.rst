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

Checked patching
----------------

A patch produced on one process may arrive after the destination structure
has already changed. The :func:`.checked_patch` function verifies every
operation against the current state before applying anything: ``change`` and
``remove`` operations must still find their recorded old value, and ``add``
operations must target an existing container that does not hold the new
key, index or item yet.

.. code-block:: python

    from dictdiffer import checked_patch

    destination = {'title': 'hello', 'tags': ['a', 'b']}
    patch_result = [
        ('change', 'title', ('hello', 'hi')),
        ('remove', 'tags', [(1, 'b')]),
    ]

    patched, conflicts = checked_patch(patch_result, destination)

    assert conflicts == []
    assert patched == {'title': 'hi', 'tags': ['a']}
    # The original destination is never modified in the default copy mode.
    assert destination == {'title': 'hello', 'tags': ['a', 'b']}

When an operation is stale, *all* locatable conflicts are returned together
and the destination stays untouched. Each
:class:`~dictdiffer.conflict.PatchConflict` exposes the zero based
``operation`` number, the canonical ``path`` as a tuple of literal path
components and a human readable ``reason``. A missing value is reported as
``<missing>`` rather than ``None``, because ``None`` can be a real value.

.. code-block:: python

    patched, conflicts = checked_patch(
        [('change', 'title', ('stale', 'hi'))], destination)

    assert patched is None
    assert conflicts[0].operation == 0
    assert conflicts[0].path == ('title',)

Nodes given as lists or tuples are interpreted literally, so dictionary
keys containing dots are never split by the dot notation. List indices are
evaluated against the state at the moment each operation runs, therefore
earlier insertions and removals are already visible to later operations.
Passing ``in_place=True`` applies the validated patch directly to the
destination, but a failed check still leaves it unchanged.

The check works on a private deep copy, so a custom mapping whose mutating
method raises cannot be rolled back automatically. In that situation
:class:`~dictdiffer.PatchApplyError` is raised and dictdiffer does not
promise to undo arbitrary external side effects performed by the custom
container.

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
