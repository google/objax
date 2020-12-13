objax.functional package
========================

.. currentmodule:: objax.functional

.. contents::
    :local:
    :depth: 1

objax.functional
----------------

.. currentmodule:: objax.functional

Due to the large number of APIs in this section, we organized it into the following sub-sections:

.. contents::
    :local:
    :depth: 1

Activation
^^^^^^^^^^

.. autosummary::

    celu
    elu
    leaky_relu
    log_sigmoid
    log_softmax
    logsumexp
    relu
    selu
    sigmoid
    softmax
    softplus
    tanh

.. autofunction:: celu
.. autofunction:: elu
.. autofunction:: leaky_relu
.. autofunction:: log_sigmoid
.. autofunction:: log_softmax
.. autofunction:: logsumexp
.. autofunction:: relu
.. autofunction:: selu
.. autofunction:: sigmoid
.. autofunction:: softmax
.. autofunction:: softplus
.. autofunction:: tanh

Pooling
^^^^^^^

.. autosummary::

    average_pool_2d
    batch_to_space2d
    channel_to_space2d
    max_pool_2d
    space_to_batch2d
    space_to_channel2d

.. autofunction:: average_pool_2d

    For a definition of pooling, including examples see
    `Pooling Layer <https://cs231n.github.io/convolutional-networks/#pool>`_.

.. autofunction:: batch_to_space2d
.. autofunction:: channel_to_space2d
.. autofunction:: max_pool_2d

    For a definition of pooling, including examples see
    `Pooling Layer <https://cs231n.github.io/convolutional-networks/#pool>`_.

.. autofunction:: space_to_batch2d
.. autofunction:: space_to_channel2d

Misc
^^^^

.. autosummary::

    dynamic_slice
    flatten
    interpolate
    one_hot
    pad
    stop_gradient
    top_k
    rsqrt
    upsample_2d
    upscale_nn

.. autofunction:: dynamic_slice
.. autofunction:: flatten
.. autofunction:: interpolate
.. autofunction:: one_hot
.. autofunction:: pad
.. autofunction:: stop_gradient
.. autofunction:: top_k
.. autofunction:: rsqrt
.. autofunction:: upsample_2d
.. autofunction:: upscale_nn

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
    mean_absolute_error
    mean_squared_error
    mean_squared_log_error
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

.. autofunction:: mean_absolute_error

.. autofunction:: mean_squared_error

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

   
