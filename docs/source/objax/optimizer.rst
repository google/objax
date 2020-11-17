objax.optimizer package
=======================

.. currentmodule:: objax.optimizer

.. autosummary::

    Adam
    ExponentialMovingAverage
    LARS
    Momentum
    SGD

.. autoclass:: Adam
    :members:

    Adam is an adaptive learning rate optimization algorithm originally presented
    in `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_.
    Specifically, when optimizing a loss function :math:`f` parameterized by model weights :math:`w`, the update
    rule is as follows:

    .. math::
      \begin{eqnarray}
      v_{k} &=& \beta_1 v_{k-1} + (1 - \beta_1) \nabla f (.; w_{k-1}) \nonumber \\
      s_{k} &=& \beta_2 s_{k-1} - (1 - \beta_2) (\nabla f (.; w_{k-1}))^2 \nonumber \\
      \hat{v_{k}} &=& \frac{v_{k}}{(1 - \beta_{1}^{k})}  \nonumber \\
      \hat{s_{k}} &=& \frac{s_{k}}{(1 - \beta_{2}^{k})} \nonumber \\
      w_{k} &=& w_{k-1} - \eta \frac{\hat{v_{k}}}{\sqrt{\hat{s_{k}}} + \epsilon} \nonumber
      \end{eqnarray}

    Adam updates exponential moving averages of the gradient :math:`(v_{k})`
    and the squared gradient :math:`(s_{k})` where the hyper-parameters
    :math:`\beta_1` and :math:`\beta_2 \in [0, 1)` control the exponential
    decay rates of these moving averages.
    The :math:`\eta` constant in the weight update rule is the learning rate and is passed
    as a parameter in the :code:`__call__` method.
    Note that the implementation uses the approximation
    :math:`\sqrt{(\hat{s_{k}} + \epsilon)} \approx \sqrt{\hat{s_{k}}} + \epsilon`.

.. autoclass:: ExponentialMovingAverage
    :members:

    When training a model, it is often beneficial to maintain exponential moving averages (EMA) of the trained
    parameters.
    Evaluations that use averaged parameters sometimes produce significantly better results than the final trained
    values (see `Acceleration of Stochastic Approximation by Averaging <http://www.meyn.ece.ufl.edu/archive/spm_files/Courses/ECE555-2011/555media/poljud92.pdf>`_).

    This maintains an EMA of the parameters passed in the VarCollection :code:`vc`.
    The EMA update rule for weights :math:`w`, the EMA :math:`m` at step :math:`t` when using a momentum
    :math:`\mu` is:

    .. math::

        m_t = \mu m_{t-1} + (1 - \mu) w_t

    The EMA weights :math:`\hat{w_t}` are simply :math:`m_t` when :code:`debias=False`.
    When :code:`debias=True`, the EMA weights are defined as:

    .. math::

        \hat{w_t} = \frac{m_t}{1 - (1 - \epsilon)\mu^t}

    Where :math:`\epsilon` is a small constant to avoid a divide-by-0.

.. autoclass:: LARS
    :members:

    The Layer-Wise Rate Scaling (LARS) optimizer implements the scheme originally proposed in
    `Large Batch Training of Convolutional Networks <https://arxiv.org/abs/1708.03888>`_. The
    optimizer takes as input the base learning rate :math:`\gamma_0`, momentum :math:`m`,
    weight decay :math:`\beta`, and trust coefficient :math:`\eta` and updates the model weights
    :math:`w` as follows:

    .. math::
      \begin{eqnarray}
      g_{t}^{l} &\leftarrow& \nabla L(w_{t}^{l}) \nonumber \\
      \gamma_t &\leftarrow& \gamma_0 \ast (1 - \frac{t}{T})^{2} \nonumber \\
      \lambda^{l} &\leftarrow& \frac{\| w_{t}^{l} \| }{ \| g_t^{l} \| + \beta \| w_{t}^{l} \|} \nonumber \\
      v_{t+1}^{l} &\leftarrow& m v_{t}^{l} + \gamma_{t+1} \ast \lambda^{l} \ast (g_{t}^{l} + \beta w_{t}^{l}) \nonumber \\
      w_{t+1}^{l} &\leftarrow& w_{t}^{l} - v_{t+1}^{l} \nonumber \\
      \end{eqnarray}

    where :math:`T` is the total number of steps (epochs) that the optimizer will take, :math:`t` is the
    current step number, and :math:`w_{t}^{l}` are the weights for during step :math:`t` for layer :math:`l`.
	
.. autoclass:: Momentum
    :members:

    The momentum optimizer (`expository article <https://distill.pub/2017/momentum/>`_) introduces a tweak to the
    standard gradient descent.
    Specifically, when optimizing a loss function :math:`f` parameterized by model weights :math:`w` the update rule
    is as follows:

    .. math::
      \begin{eqnarray}
      v_{k} &=& \mu v_{k-1} + \nabla f (.; w_{k-1}) \nonumber \\
      w_{k} &=& w_{k-1} - \eta v_{k} \nonumber
      \end{eqnarray}

    The term :math:`v` is the *velocity*: It accumulates past gradients through a weighted moving average calculation.
    The parameters :math:`\mu, \eta` are the *momentum* and the *learning rate*.

    The momentum class also implements Nesterov's Accelerated Gradient (NAG)
    (see `Sutskever et. al. <http://proceedings.mlr.press/v28/sutskever13.pdf>`_).
    Like momentum, NAG is a first-order optimization method with better convergence
    rate than gradient descent in certain situations. The NAG update can be
    written as:

    .. math::
      \begin{eqnarray}
      v_{k} &=& \mu v_{k-1} + \nabla f(.; w_{k-1} + \mu v_{k-1}) \nonumber \\
      w_{k} &=& w_{k-1} - \eta v_{k} \nonumber
      \end{eqnarray}

    The implementation uses the simplification presented by `Bengio et. al
    <https://arxiv.org/pdf/1212.0901v2.pdf>`_.

.. autoclass:: SGD
    :members:

    The stochastic gradient optimizer performs
    `Stochastic Gradient Descent <https://en.wikipedia.org/wiki/Stochastic_gradient_descent>`_ (SGD).
    It uses the following update rule for a loss :math:`f` parameterized with model weights :math:`w` and
    a user provided learning rate :math:`\eta`:

    .. math::

        w_k = w_{k-1} - \eta\nabla f(.; w_{k-1})
