objax.functional package
========================

.. currentmodule:: objax.functional

.. contents::
    :local:
    :depth: 1

objax.functional
----------------

.. currentmodule:: objax.functional

.. autosummary::

   average_pool_2d
   flatten
   max_pool_2d
   one_hot
   relu
   upscale

.. autofunction:: average_pool_2d

    For a definition of pooling, including examples see
    `Pooling Layer <https://cs231n.github.io/convolutional-networks/#pool>`_.

.. autofunction:: max_pool_2d

    For a definition of pooling, including examples see
    `Pooling Layer <https://cs231n.github.io/convolutional-networks/#pool>`_.

.. autofunction:: flatten

.. autofunction:: one_hot

.. autofunction:: relu

.. autofunction:: upscale

objax.functional.divergence
---------------------------

.. currentmodule:: objax.functional.divergence

.. autosummary::

   kl

.. autofunction:: kl

   .. math::
      kl(p,q) = p \cdot \log{\frac{p + \epsilon}{q + \epsilon}}

   The :math:`\epsilon` term is added to ensure that neither :code:`p` nor :code:`q` are zero.

objax.functional.loss
---------------------

.. currentmodule:: objax.functional.loss

.. autosummary::

    cross_entropy_logits
    cross_entropy_logits_sparse
    l2
    sigmoid_cross_entropy_logits

.. autofunction:: cross_entropy_logits

Calculates the cross entropy loss, defined as follows: 

   .. math::

      \begin{eqnarray}
      l(y,\hat{y}) & = & - \sum_{j=1}^{q} y_j \log \frac{e^{o_j}}{\sum_{k=1}^{q} e^{o_k}} \nonumber \\
      & = & \log \sum_{k=1}^{q} e^{o_k} - \sum_{j=1}^{q} y_j o_j \nonumber
      \end{eqnarray}

   where :math:`o_k` are the logits and :math:`y_k` are the labels. 

.. autofunction:: cross_entropy_logits_sparse

.. autofunction:: l2

Calculates the l2 loss, as:

.. math::
   l_2 = \frac{\sum_{i} x_{i}^2}{2}

.. autofunction:: sigmoid_cross_entropy_logits


objax.functional.parallel
-------------------------

.. currentmodule:: objax.functional.parallel

.. autosummary::

   pmax
   pmean
   pmin
   psum

.. autofunction:: pmax
.. autofunction:: pmean
.. autofunction:: pmin
.. autofunction:: psum

   
