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

Composing patches
=================

If you store only the net change between two states but happen to hold two
consecutive patches (``A -> B`` and ``B -> C``), :func:`.compose` merges them
into a single patch that takes *A* directly to *C*. The intermediate document
*B* is not required - only the values carried by the two patches are used.

.. code-block:: python

    from dictdiffer import compose

    first = {'a': 1, 'l': [1, 2], 's': {1}}
    middle = {'a': 2, 'l': [1, 2, 3], 's': {1, 2}}
    third = {'a': 2, 'l': [1, 2, 3], 's': {1, 2}}

    net = compose(diff(first, middle), diff(middle, third))

    assert patch(net, first) == third

The composition is purely symbolic and minimal:

* an addition that is later deleted (or a modification that is later
  reverted) is dropped from the net patch;
* list insertions and deletions are merged while compensating for the index
  drift they cause (a delete followed by an append at the freed position
  becomes a ``change``);
* ``set`` add/remove operations are merged element by element;
* dictionary edits at arbitrary sub-paths are combined recursively.

The resulting patch keeps the original *A* values on every ``change`` and
``remove`` item, so it can be reverted to recover *A*:

.. code-block:: python

    assert revert(net, third) == first

When the second patch assumes an old value or a structure that the first
patch does not leave behind (e.g. it modifies a key that was removed, or
expects another old value), :class:`.ComposeConflict` is raised before any
partial result is produced, instead of failing halfway with a ``KeyError``.

Patches produced by ``diff`` (including the ``expand=True`` and
``path_limit`` forms) are supported. The only inherent ambiguity is the
single ``('add', node, [(0, a_set)])`` tuple, which cannot distinguish a set
mutation from an empty list that receives one ``set`` element; compose treats
it as a set mutation unless the other patch provides evidence to the
contrary.

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
