Adding or changing Objax modules
================================

This guide explains how to add a new module to Objax or change existing one.

In addition to this guide, consider looking at an `example pull request <https://github.com/google/objax/pull/43/files>`_
which adds new module with documentation.

Writing code
------------

When adding new module or function, you have to decide where to put it.
Typical locations of new modules and functions are the following:

* `objax/functional <https://github.com/google/objax/tree/master/objax/functional>`_ contains
  various stateless functions (non-modules) which are used in machine learning.
  For example: loss functions, activations, stateless ops.
* `objax/io <https://github.com/google/objax/tree/master/objax/io>`_ contains routines for model saving and loading.
* `objax/nn <https://github.com/google/objax/tree/master/objax/nn>`_ contains layers,
  which serve as building blocks for neural network. It also contains initializes for layer parameters. 
* `objax/optimizer <https://github.com/google/objax/tree/master/objax/optimizer>`_ contains optimizers.
* `objax/privacy <https://github.com/google/objax/tree/master/objax/privacy>`_ contains code
  for privacy-preserving training of neural networks.
* `objax/random <https://github.com/google/objax/tree/master/objax/random>`_ contains routines
  for random number generation.
* `objax/zoo <https://github.com/google/objax/tree/master/objax/zoo>`_ is a "model zoo"
  of various well-known neural network architectures.

When writing code we follow PEP8 style guide with the following two exceptions:

* We maximum line length to 120 characters.
* We allow to assign lambda, e.g. :code:`f = lambda x: x`

Script :code:`./tests/run_linter.sh` automatically checks majority of code style violations.
`PyCharm code formatter <https://www.jetbrains.com/help/pycharm/command-line-formatter.html>`_
could be used to automatically reformat code.

Writing unit tests
------------------

Unit tests are required for all code changes and new code of Objax library.
However test are not required for `examples <https://github.com/google/objax/tree/master/examples>`_.

All unit tests are placed into `tests <https://github.com/google/objax/tree/master/tests>`_ directory.
They are grouped into different files based on what they are testing.
For example `tests/conv.py <https://github.com/google/objax/blob/master/tests/conv.py>`_
contains unit tests for convolution modules.

We use `Python unittest <https://docs.python.org/3/library/unittest.html>`_ module for tests.

Writing documentation
---------------------

Documentation for specific APIs is written inside docstrings within the code
(`example for Conv2D <https://github.com/google/objax/blob/ae09d05aab2964912fdcecb7e3be31a2aca6079f/objax/nn/layers.py#L151>`_).

Other documentation is stored in `docs <https://github.com/google/objax/tree/master/docs>`_
subdirectory of Objax repository.
It uses `reST <https://docutils.sourceforge.io/rst.html>`_ as a markup language,
and `Sphinx <https://www.sphinx-doc.org/>`_ automatically generates documentation
for `<objax.readthedocs.io>`_.

Docstrings
^^^^^^^^^^

All public facing classes, functions and class methods
should have a short `docstring <https://www.python.org/dev/peps/pep-0257>`_
describing what they are doing.
Functions and methods should also have a description of their arguments and
return value.

To keep code easy to read, we recommend to write short and concise docstrings:

* Try to keep description of classes and functions with 1-5 lines.
* Try to fit description of each argument of each function within 1-2 lines.
* Avoid putting examples or long descriptions into docstrings, those should go
  into `reST` docs.

Here is an example of how to write docstring for a function:

.. code-block:: python

    def cross_entropy_logits(logits: JaxArray, labels: JaxArray) -> JaxArray:
        """Computes the cross-entropy loss.

        Args:
            logits: (batch, #class) tensor of logits.
            labels: (batch, #class) tensor of label probabilities (e.g. labels.sum(axis=1) must be 1)

        Returns:
            (batch,) tensor of the cross-entropies for each entry.
        """
        return logsumexp(logits, axis=1) - (logits * labels).sum(1)

If you only updating existing docstrings these changes will be automatically reflected
in `<objax.readthedocs.io>`_ after pull request is merged into repository.
When adding docstrings for new classes and functions, you also may need to
update `reST` files as described below.

reST documentation
^^^^^^^^^^^^^^^^^^

Updates of `reST` files are required either when new APIs are added (new function, new module)
or when other (non API) documentation is needed.

Most of the API documentation is located in `docs/source/objax <https://github.com/google/objax/tree/master/docs/source/objax>`_
They are grouped into different `.rst` files by the name of python package,
for example `docs/source/objax/nn.rst` contains documentation for `objax.nn` package.

To add new class or function, you typically need to add name of the class or function into :code:`autosummary` section
and add :code:`autoclass` or :code:`autofunction` section for new class/function.
Here is an example of changes which are needed to add documentation for :code:`Conv2D` module:

.. code-block:: rst

    .. autosummary::

      ...
      Conv2D
      ...

    ...

    .. autoclass:: Conv2D
        :members:

        Additional documentation (non-docstrings) for Conv2D goes here.

For reference about `reST` syntax, refer to
`reST documentation <https://docutils.sourceforge.io/rst.html>`_
or `cheat sheet <http://openalea.gforge.inria.fr/doc/openalea/doc/_build/html/source/sphinx/rest_syntax.html>`_.
