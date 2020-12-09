Frequently Asked Questions
==========================

What is the difference between Objax and other JAX frameworks?
--------------------------------------------------------------

JAX itself as well as most of JAX frameworks (other than Objax)
follows a functional style programming paradigm.
This means that all computations are expected to be performed by
stateless `pure functions <https://en.wikipedia.org/wiki/Pure_function>`_.
And state (i.e. model weights) has to be manually passed to these functions.

On the other hand, Objax follows an object-oriented programming paradigm
(similar to PyTorch and Tensorflow).
Objax provides objects (called Objax modules) which store and manage
the state of a neural network.

To better illustrate this distinction,
below are two examples of a similar code written in pure JAX and Objax.

Every time when a user calls neural network components in JAX (and many JAX frameworks),
they have to pass both neural network parameters :code:`params`
as well as training examples :code:`batch['x'], batch['y']`::

  params = (jn.zeros(ndim), jn.zeros(1))

  def loss(params, x, y):
      w, b = params
      pred = jn.dot(x, w) + b
      return 0.5 * ((y - pred) ** 2).mean()

  g_fn = jax.grad(loss)              # g_fn is a function

  # Need to pass both parameters and batch to g_fn
  g_value = g_fn(params, batch['x'], batch['y'])

On the other, modules in Objax store parameters and state internally.
Thus a user only has to pass around training examples :code:`batch['x'], batch['y']`::

  w = objax.TrainVar(jn.zeros(ndim))
  b = objax.TrainVar(jn.zeros(1))

  def loss(x, y):
      pred = jn.dot(x, w.value) + b.value
      return 0.5 * ((y - pred) ** 2).mean()

  g_fn = objax.Grad(loss,           # g_fn is Objax module
                    objax.VarCollection({'w': w, 'b': b}))

  # Need to pass only batch to g_fn
  g_value = g_fn(batch['x'], batch['y'])

What is the difference between Objax and PyTorch/Tensorflow?
------------------------------------------------------------

Execution runtime
^^^^^^^^^^^^^^^^^

Objax is implemented on top of JAX,
while PyTorch and Tensorflow have their own underlying runtime environments.
In practice it mainly means that to interoperate between these frameworks
some conversion needs to be done.
For example convert PyTorch/Tensorflow tensor to NumPy array
and then feed this NumPy array to code in Objax.

Design of API
^^^^^^^^^^^^^

Objax was inspired by the best of other machine learning frameworks
(including PyTorch and Tensorflow).
Thus readers may observe similarities between Objax API and API of PyTorch
(or some other frameworks).

Nevertheless, **Objax is not intended to be a re-implementation of the API
of any other framework and each Objax design decision is weighted on its own merit**.
So there will always be differences between Objax API and APIs of other frameworks.
