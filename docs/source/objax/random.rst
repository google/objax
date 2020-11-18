objax.random package
====================

.. currentmodule:: objax.random

.. autosummary::

    Generator
    normal
    randint
    truncated_normal
    uniform

.. autoclass:: Generator
    :members: __init__, seed, __call__, key

    The default generator can be accessed through :code:`objax.random.DEFAULT_GENERATOR`.
    Its seed is **0** by default, and can be set through :code:`objax.random.DEFAULT_GENERATOR.seed(s)`
    where integer **s** is the desired seed.

.. autofunction:: normal
.. autofunction:: randint
.. autofunction:: truncated_normal
.. autofunction:: uniform
