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

Two consecutive patches can be composed into one without keeping the
intermediate object with :func:`.compose`.  It only inspects the patches,
never fabricates a base object, raises :class:`~dictdiffer.ComposeConflict`
when the second patch contradicts the intermediate value, and raises
:class:`~dictdiffer.ComposeUnknown` when a pre-condition cannot be derived
from the patches (the caller may supply those few values through ``prior``):

.. code-block:: python

    from dictdiffer import compose

    # A -> B and B -> C patches, archived as a single A -> C patch.
    p = [('change', 'title', ('hello', 'hello!'))]
    q = [('change', 'title', ('hello!', 'hello!!'))]
    composed = list(compose(p, q))

    assert composed == [('change', 'title', ('hello', 'hello!!'))]
    assert patch(composed, first) == patch(q, patch(p, first))
    assert revert(composed, patch(composed, first)) == first

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
