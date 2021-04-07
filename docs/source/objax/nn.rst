objax.nn package
================

objax.nn
--------

.. currentmodule:: objax.nn

.. autosummary::

   BatchNorm
   BatchNorm0D
   BatchNorm1D
   BatchNorm2D
   Conv2D
   ConvTranspose2D
   Dropout
   Linear
   MovingAverage
   ExponentialMovingAverage
   Sequential
   SyncedBatchNorm
   SyncedBatchNorm0D
   SyncedBatchNorm1D
   SyncedBatchNorm2D

.. autoclass:: BatchNorm
   :members:

   .. math::
      y = \frac{x-\mathrm{E}[x]}{\sqrt{\mathrm{Var}[x]+\epsilon}} \times \gamma + \beta

   The mean (:math:`\mathrm{E}[x]`) and variance (:math:`\mathrm{Var}[x]`) are calculated per specified dimensions
   and over the mini-batches.
   :math:`\beta` and :math:`\gamma` are trainable parameter tensors of shape **dims**.
   The elements of :math:`\beta` are initialized with zeros and those of :math:`\gamma` are initialized with ones.

.. autoclass:: BatchNorm0D
    :members: __call__

    .. math::
      y = \frac{x-\mathrm{E}[x]}{\sqrt{\mathrm{Var}[x]+\epsilon}} \times \gamma + \beta

    The mean (:math:`\mathrm{E}[x]`) and variance (:math:`\mathrm{Var}[x]`) are calculated over the mini-batches.
    :math:`\beta` and :math:`\gamma` are trainable parameter tensors of shape (1, **nin**).
    The elements of :math:`\beta` are initialized with zeros and those of :math:`\gamma` are initialized with ones.

.. autoclass:: BatchNorm1D
    :members: __call__

    .. math::
      y = \frac{x-\mathrm{E}[x]}{\sqrt{\mathrm{Var}[x]+\epsilon}} \times \gamma + \beta

    The mean (:math:`\mathrm{E}[x]`) and variance (:math:`\mathrm{Var}[x]`) are calculated per channel and over
    the mini-batches.
    :math:`\beta` and :math:`\gamma` are trainable parameter tensors of shape (1, **nin**, 1).
    The elements of :math:`\beta` are initialized with zeros and those of :math:`\gamma` are initialized with ones.


.. autoclass:: BatchNorm2D
    :members: __call__

    .. math::
      y = \frac{x-\mathrm{E}[x]}{\sqrt{\mathrm{Var}[x]+\epsilon}} \times \gamma + \beta

    The mean (:math:`\mathrm{E}[x]`) and variance (:math:`\mathrm{Var}[x]`) are calculated per channel and over
    the mini-batches. :math:`\beta` and :math:`\gamma` are trainable parameter tensors of shape (1, **nin**, 1, 1).
    The elements of :math:`\beta` are initialized with zeros and those of :math:`\gamma` are initialized with ones.


.. autoclass:: Conv2D
    :members:

    In the simplest case (**strides** = 1, **padding** = VALID), the output tensor
    :math:`(N,C_{out},H_{out},W_{out})` is computed from an input tensor :math:`(N,C_{in},H,W)`
    with kernel weight :math:`(k,k,C_{in},C_{out})` and bias :math:`(C_{out})` as follows:

    .. math::
      \mathrm{out}[n,c,h,w] = \mathrm{b}[c] + \sum_{t=0}^{C_{in}-1}\sum_{i=0}^{k-1}\sum_{j=0}^{k-1} \mathrm{in}[n,c,i+h,j+w] \times \mathrm{w}[i,j,t,c]

    where :math:`H_{out}=H-k+1`, :math:`W_{out}=W-k+1`.
    Note that the implementation follows the definition of
    `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_.
    When **padding** = SAME, the input tensor is zero-padded by :math:`\lfloor\frac{k-1}{2}\rfloor` for left and
    up sides and :math:`\lfloor\frac{k}{2}\rfloor` for right and down sides.

.. autoclass:: ConvTranspose2D
   :members:

.. autoclass:: Dropout
    :members:

    During the evaluation, the module does not modify the input tensor.
    Dropout (`Improving neural networks by preventing co-adaptation of feature detectors <https://arxiv.org/abs/1207.0580>`_)
    is an effective regularization technique which reduces the overfitting and increases the overall utility.

.. autoclass:: Linear
    :members:

    The output tensor :math:`(N,C_{out})` is computed from an input tensor :math:`(N,C_{in})` with kernel weight
    :math:`(C_{in},C_{out})` and bias :math:`(C_{out})` as follows:

    .. math::
      \mathrm{out}[n,c] = \mathrm{b}[c] + \sum_{t=1}^{C_{in}} \mathrm{in}[n,t] \times \mathrm{w}[t,c]

.. autoclass:: MovingAverage
   :members:

.. autoclass:: ExponentialMovingAverage
   :members:

   .. math::
      x_{\mathrm{EMA}} \leftarrow \mathrm{momentum} \times x_{\mathrm{EMA}} + (1-\mathrm{momentum}) \times x

.. autoclass:: Sequential
   :members: __init__, append, clear, copy, count, extend, index, insert, pop, remove, reverse, vars

    Usage example::

        import objax

        ml = objax.nn.Sequential([objax.nn.Linear(2, 3), objax.functional.relu,
                                  objax.nn.Linear(3, 4)])
        x = objax.random.normal((10, 2))
        y = ml(x)  # Runs all the operations (Linear -> ReLU -> Linear).
        print(y.shape)  # (10, 4)

        # objax.nn.Sequential is really a list.
        ml.insert(2, objax.nn.BatchNorm0D(3))  # Add a batch norm layer after ReLU
        ml.append(objax.nn.Dropout(keep=0.5))  # Add a dropout layer at the end
        y = ml(x, training=False)  # Both batch norm and dropout expect a training argument.
        # Sequential automatically pass arguments to the modules using them.

        # You can run a subset of operations since it is a list.
        y1 = ml[:2](x)  # Run first two layers (Linear -> ReLU)
        y2 = ml[2:](y1, training=False)  # Run all layers starting from third (BatchNorm0D -> Dropout)
        print(ml(x, training=False) - y2)  # [[0. 0. ...]] - results are the same.

        print(ml.vars())
        # (Sequential)[0](Linear).b                              3 (3,)
        # (Sequential)[0](Linear).w                              6 (2, 3)
        # (Sequential)[2](BatchNorm0D).running_mean              3 (1, 3)
        # (Sequential)[2](BatchNorm0D).running_var               3 (1, 3)
        # (Sequential)[2](BatchNorm0D).beta                      3 (1, 3)
        # (Sequential)[2](BatchNorm0D).gamma                     3 (1, 3)
        # (Sequential)[3](BatchNorm0D).running_mean              3 (1, 3)
        # (Sequential)[3](BatchNorm0D).running_var               3 (1, 3)
        # (Sequential)[3](BatchNorm0D).beta                      3 (1, 3)
        # (Sequential)[3](BatchNorm0D).gamma                     3 (1, 3)
        # (Sequential)[4](Linear).b                              4 (4,)
        # (Sequential)[4](Linear).w                             12 (3, 4)
        # (Sequential)[5](Dropout).keygen(Generator)._key        2 (2,)
        # +Total(13)                                            51

.. autoclass:: SyncedBatchNorm
    :members: __call__

.. autoclass:: SyncedBatchNorm0D
    :members: __call__

.. autoclass:: SyncedBatchNorm1D
    :members: __call__

.. autoclass:: SyncedBatchNorm2D
    :members: __call__

objax.nn.init
-------------

.. currentmodule:: objax.nn.init

.. autosummary::
    gain_leaky_relu
    identity
    kaiming_normal_gain
    kaiming_normal
    kaiming_truncated_normal
    orthogonal
    truncated_normal
    xavier_normal_gain
    xavier_normal
    xavier_truncated_normal

.. autoclass:: gain_leaky_relu
   :members:

   The returned gain value is

   .. math::
    \sqrt{\frac{2}{1 + \text{relu_slope}^2}}.

.. autoclass:: identity
   :members:

.. autoclass:: kaiming_normal_gain
   :members:

   The returned gain value is

   .. math::
    \sqrt{\frac{1}{\text{fan_in}}}.

.. autoclass:: kaiming_normal
   :members:

.. autoclass:: kaiming_truncated_normal
   :members:

.. autoclass:: orthogonal
   :members:

.. autoclass:: truncated_normal
   :members:

.. autoclass:: xavier_normal_gain
   :members:

   The returned gain value is

   .. math::
    \sqrt{\frac{2}{\text{fan_in} + \text{fan_out}}}.

.. autoclass:: xavier_normal
   :members:

.. autoclass:: xavier_truncated_normal
   :members:

