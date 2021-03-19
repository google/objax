Compilation and Parallelism
===========================

In this section we discuss the concepts of code compilation and parallelism typically for the purpose of accelerated
performance.
We'll cover the following subtopics:

.. contents::
    :local:
    :depth: 3


* Just-In-Time (JIT) Compilation is a compilation of the code on the first time it’s executed with the goal of
  speeding up subsequent runs.
* Parallelism runs operations on multiple devices (for example multiple GPUs).
* Vectorization can be seen as batch-level parallelization, running an operation on a batch in parallel.


JIT Compilation
---------------

:py:class:`objax.Jit` is a Module that takes a module or a function and compiles it for faster performance.

As a simple starting example, let's jit a module::

    import objax

    net = objax.nn.Sequential([objax.nn.Linear(2, 3), objax.functional.relu, objax.nn.Linear(3, 4)])
    jit_net = objax.Jit(net)

    x = objax.random.normal((100, 2))
    print(((net(x) - jit_net(x)) ** 2).sum())  # 0.0

You can also jit a function, in this case you must decorate the function with the variables it uses::

    @objax.Function.with_vars(net.vars())
    def my_function(x):
        return objax.functional.softmax(net(x))

    jit_func = objax.Jit(my_function)
    print(((objax.functional.softmax(net(x)) - jit_func(x)) ** 2).sum())

In terms of performance, on this small example there's a significant gain in speed, numbers vary depending on hardware
present in your computer and what code is being jitted::

    from time import time

    t0 = time(); y = net(x); print(time() - t0)       # 0.005...
    t0 = time(); y = jit_net(x); print(time() - t0)   # 0.001...

As mentioned earlier, :code:`jit_net` is a module instance, it's sharing the variables with the module :code:`net`,
we can verify it::

    print(net.vars())
    # (Sequential)[0](Linear).b        3 (3,)
    # (Sequential)[0](Linear).w        6 (2, 3)
    # (Sequential)[2](Linear).b        4 (4,)
    # (Sequential)[2](Linear).w       12 (3, 4)
    # +Total(4)                       25

    print(jit_net.vars())
    # (Jit)(Sequential)[0](Linear).b        3 (3,)
    # (Jit)(Sequential)[0](Linear).w        6 (2, 3)
    # (Jit)(Sequential)[2](Linear).b        4 (4,)
    # (Jit)(Sequential)[2](Linear).w       12 (3, 4)
    # +Total(4)                            25

    # We can verify that jit_func also shares the same variables
    print(jit_func.vars())
    # (Jit){my_function}(Sequential)[0](Linear).b        3 (3,)
    # (Jit){my_function}(Sequential)[0](Linear).w        6 (2, 3)
    # (Jit){my_function}(Sequential)[2](Linear).b        4 (4,)
    # (Jit){my_function}(Sequential)[2](Linear).w       12 (3, 4)
    # +Total(4)                                         25

Note that we only verified that the variables names and sizes were the same (or almost the same since the variables
in Jit are prefixed with :code:`(Jit)`).
Let's now verify that the weights are indeed shared by modifying the weights::

    net[-1].b.assign(net[-1].b.value + 1)
    print(((net(x) - jit_net(x)) ** 2).sum())  # 0.0
    # Both net(x) and jit_net(x) were affected in the same way by the change
    # since the weights are shared.
    # You can also inspect the values print(net(x)) for more insight.

.. _jitted-train-label:

A realistic case: Fully jitted training step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's write a classifier training op, this is very similar to example shown in :ref:`loss-optimization-label`.
We are going to define a model, an optimizer, a loss and a gradient::

    import objax

    m = objax.nn.Sequential([objax.nn.Linear(2, 3), objax.functional.relu, objax.nn.Linear(3, 4)])
    opt = objax.optimizer.Momentum(m.vars())

    @objax.Function.with_vars(m.vars())
    def loss(x, labels):
        logits = m(x)
        return objax.functional.loss.cross_entropy_logits(logits, labels).mean()

    gradient_loss = objax.GradValues(loss, m.vars())

    @objax.Function.with_vars(m.vars() + opt.vars())
    def train(x, labels, lr):
        g, v = gradient_loss(x, labels)  # Compute gradients and loss
        opt(lr, g)                       # Apply SGD
        return v                         # Return loss value

    # It's better to jit the top level call to allow internal optimizations.
    train_jit = objax.Jit(train)

Note that we passed to Jit all the vars that were used in :code:`train`.
We passed :code:`gradient_loss.vars() + opt.vars()`.
Why didn't we pass :code:`m.vars() + gradient_loss.vars() + opt.vars()`?
We could and it's perfectly fine to do so, but keep in mind that :code:`gradient_loss` is itself a module which shares
the weights of :code:`m` and consequently :code:`m.vars()` is already included in :code:`gradient_loss.vars()`.

Static arguments
^^^^^^^^^^^^^^^^

Static arguments are arguments that are treated as static (compile-time constant) in the jitted function.
Boolean arguments, numerical arguments used in comparisons (resulting in a bool), strings must be marked as static.

Calling the jitted function with different values for these constants will trigger recompilation.
As a rule of thumb:

* Good static arguments: training (boolean), my_mode (int that can take only a few values), ...
* Bad static arguments: training_step (int that can take a lot of values)

Let's look at an example with BatchNorm which takes a training argument:

.. code-block:: python
    :emphasize-lines: 9

    import objax

    net = objax.nn.Sequential([objax.nn.Linear(2, 3), objax.nn.BatchNorm0D(3)])

    @objax.Function.with_vars(net.vars())
    def f(x, training):
        return net(x, training=training)

    jit_f_static = objax.Jit(f, static_argnums=(1,))
    # Note the static_argnums=(1,) which indicates that argument 1 (training) is static.

    x = objax.random.normal((100, 2))
    print(((net(x, training=False) - jit_f_static(x, False)) ** 2).sum())  # 0.0

What happens if you don't use :code:`static_argnums`?

.. code-block:: python
    :emphasize-lines: 3-9

    jit_f = objax.Jit(f)
    y = jit_f(x, False)
    # Traceback (most recent call last):
    #   File <...>
    #   <long stack trace>
    # jax.core.ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected (in `bool`).
    # Use transformation parameters such as `static_argnums` for `jit` to avoid tracing input values.
    # See `https://jax.readthedocs.io/en/latest/faq.html#abstract-tracer-value-encountered-where-concrete-value-is-expected-error`.
    # Encountered value: Traced<ShapedArray(bool[], weak_type=True):JaxprTrace(level=-1/1)>

To cut a long story short: when compiling boolean inputs must be made static.


For more information, please refer to
`jax.jit <https://jax.readthedocs.io/en/latest/jax.html?highlight=jit#jax.jit>`_ which is the API Objax uses under
the hood.

Constant optimization
^^^^^^^^^^^^^^^^^^^^^

As seen previously, :py:class:`objax.Jit` takes a :code:`variables` argument to specify the variables of a function
or of a module that Jit is compiling.

If a variable is not passed to Jit it will be treated as a constant and will be optimized away.

.. warning::

    A jitted module will **not** see any change made to a constant.
    A constant is not expected to change since it is supposed to be... constant!

A simple constant optimization example::

    import objax

    m = objax.nn.Linear(3, 4)
    # Pass an empty VarCollection to signify to Jit that m has no variable.
    jit_constant = objax.Jit(m, objax.VarCollection())

    x = objax.random.normal((10, 3))
    print(((m(x) - jit_constant(x)) ** 2).sum())  # 0.0

    # Modify m (which was supposed to be constant!)
    m.b.assign(m.b.value + 1)
    print(((m(x) - jit_constant(x)) ** 2).sum())  # 40.0
    # As expected jit_constant didn't see the change.


.. warning::

    The XLA backend (the interface to the hardware) will do the constant optimization and may take a long time and
    a lot of memory due to compilation, often with very little gain in final performance, if any.

RandomState and Jit
^^^^^^^^^^^^^^^^^^^

You need to includes the generator's (e.g., :code:`objax.random.DEFAULT_GENERATOR`) variables to the variables
of the module of function that is jitted, otherwise they will be treated as constants and the jitted function
will always return the same value. 

.. code-block:: python

    import objax
    objax.random.DEFAULT_GENERATOR.seed(123)

    @objax.Function.with_vars(objax.random.DEFAULT_GENERATOR.vars())
    def function():
        d = objax.random.uniform((1,))
        return d

    function_jit = objax.Jit(function)
    for _ in range(3):
        print(function_jit())
        # Prints three different random values


Parallelism
-----------

.. note::

    If you don't have multiple devices, you can simulate them on CPU by starting python with the following command:

    .. code-block:: bash

        CUDA_VISIBLE_DEVICES= XLA_FLAGS=--xla_force_host_platform_device_count=8 python

    Alternatively you can do it in Python directly by inserting this snippet **before importing Objax**:

    .. code-block:: python

        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'


:py:class:`objax.Parallel` provides a way to distribute computations across multi-GPU (or TPU).
It also performs JIT under the hood and its API shares a lot with :py:class:`objax.Jit`:
It takes a function to be compiled, a :code:`VarCollection` as well as a
`static_argnums` parameters which all behave the same as in Jit.
However it also takes specific arguments for the task of handling parallelism which we are going to introduce.

When running a parallelized a function :code:`f` on a batch :math:`x` of shape :math:`(n, ...)` on :math:`d` devices,
the following steps happen:

1. The batch :math:`x` is divided into :math:`d` sub-batches
   :math:`x_i` of shape :math:`(n/d, ...)` for :math:`i\in\{0, ..., d-1\}`
2. Each sub-batch :math:`x_i` is passed to :code:`f` and ran on device :math:`i` in parallel.
3. The results are collected as output sub-subatches :math:`y_i=f(x_i)`
4. The outputs :math:`y_i` are represented as a single tensor :math:`y` of shape :math:`(d, ...)`
5. The final output is obtained by calling the :code:`reduce` function on :math:`y`: :code:`out = reduce(y)`.

With this in mind, we can now detail the additional arguments of :py:class:`objax.Parallel`:

* :code:`reduce`: a function that aggregates the output results from each GPU/TPU.
* :code:`axis_name`: is the name of the device dimension which we referred to as :math:`d` earlier. By default, it is
  called :code:`'device'`.

Let's illustrate this with a simple example with the parallelization of a module (:code:`para_net`) and of a function
(:code:`para_func`)::

    # This code was run on 8 simulated devices
    import objax

    net = objax.nn.Sequential([objax.nn.Linear(3, 4), objax.functional.relu])
    para_net = objax.Parallel(net)
    para_func = objax.Parallel(objax.Function(lambda x: net(x) + 1, net.vars()))

    # A batch of mockup data
    x = objax.random.normal((96, 3))

    # We're running on multiple devices, copy the model variables to all of them first.
    with net.vars().replicate():
        y = para_net(x)
        z = para_func(x)

    print(((net(x) - y) ** 2).sum())        # 8.90954e-14
    print(((net(x) - (z - 1)) ** 2).sum())  # 4.6487814e-13

We can also show the parallel version of :ref:`jitted-train-label`, highlighted are the changes from the jitted version:

.. code-block:: python
    :emphasize-lines: 16,17,20

    import objax

    m = objax.nn.Sequential([objax.nn.Linear(2, 3), objax.functional.relu, objax.nn.Linear(3, 4)])
    opt = objax.optimizer.Momentum(m.vars())

    @objax.Function.with_vars(m.vars())
    def loss(x, labels):
        logits = m(x)
        return objax.functional.loss.cross_entropy_logits(logits, labels).mean()

    gradient_loss = objax.GradValues(loss, m.vars())

    @objax.Function.with_vars(m.vars() + opt.vars())
    def train(x, labels, lr):
        g, v = gradient_loss(x, labels)                     # Compute gradients and loss
        opt(lr, objax.functional.parallel.pmean(g))        # Apply averaged gradients
        return objax.functional.parallel.pmean(v)          # Return averaged loss value

    # It's better to parallelize the top level call to allow internal optimizations.
    train_para = objax.Parallel(train, reduce=lambda x:x[0])

Let's study the concepts introduced in this example in more details.

Variable replication
^^^^^^^^^^^^^^^^^^^^

Variable replication copies the variables to multiple devices' own memory.
It is necessary to do variable replication before calling a parallelized module or function.
Variable replication is done through :py:meth:`objax.VarCollection.replicate` which is a context manager.
One could go further and creating their own replication, this is not covered here but the source of :code:`replicate` is
rather simple and a good starting point.

Here is a detailed example::

    # This code was run on 8 simulated devices
    import objax
    import jax.numpy as jn

    m = objax.ModuleList([objax.TrainVar(jn.arange(5))])
    # We use "repr" to see the whole type information.
    print(repr(m[0].value))  # DeviceArray([0, 1, 2, 3, 4], dtype=int32)

    with m.vars().replicate():
        # In the scope of the with-statement, the variables are replicated to all devices.
        print(repr(m[0].value))
        # ShardedDeviceArray([[0, 1, 2, 3, 4],
        #                     [0, 1, 2, 3, 4],
        #                     [0, 1, 2, 3, 4],
        #                     [0, 1, 2, 3, 4],
        #                     [0, 1, 2, 3, 4],
        #                     [0, 1, 2, 3, 4],
        #                     [0, 1, 2, 3, 4],
        #                     [0, 1, 2, 3, 4]], dtype=int32)
        # SharedDeviceArray is a DeviceArray across multiple devices.

    # When exiting the with-statement, the variables are reduced back to their original device.
    print(repr(m[0].value))  # DeviceArray([0., 1., 2., 3., 4.], dtype=float32)

Something interesting happened: the value of :code:`m[0]` was initially of type integer but it became a float by the
end.
This is due to the reduction that follows a replication.
By default, the reduction method takes the average of the copies on each device.
And the average of multiple integer values is casted automatically to a float.

You can customize the variable reduction, this is not something one typically would need to do but it's available for
advanced users nonetheless::

    # This code was run on 8 simulated devices
    import objax
    import jax.numpy as jn

    m = objax.ModuleList([objax.TrainVar(jn.arange(5), reduce=lambda x: x[0]),
                          objax.TrainVar(jn.arange(5), reduce=lambda x: x.sum(0)),
                          objax.TrainVar(jn.arange(5), reduce=lambda x: jn.stack(x))])
    print(repr(m[0].value))  # DeviceArray([0, 1, 2, 3, 4], dtype=int32)
    print(repr(m[1].value))  # DeviceArray([0, 1, 2, 3, 4], dtype=int32)
    print(repr(m[2].value))  # DeviceArray([0, 1, 2, 3, 4], dtype=int32)

    with m.vars().replicate():
        pass

    # When exiting the with-statement, the variables are reduced back to their original device.
    print(repr(m[0].value))  # DeviceArray([0, 1, 2, 3, 4], dtype=int32)
    print(repr(m[1].value))  # DeviceArray([ 0,  8, 16, 24, 32], dtype=int32)
    print(repr(m[2].value))  # DeviceArray([[0, 1, 2, 3, 4],
                             #              [0, 1, 2, 3, 4],
                             #              [0, 1, 2, 3, 4],
                             #              [0, 1, 2, 3, 4],
                             #              [0, 1, 2, 3, 4],
                             #              [0, 1, 2, 3, 4],
                             #              [0, 1, 2, 3, 4],
                             #              [0, 1, 2, 3, 4]], dtype=int32)


Output aggregation
^^^^^^^^^^^^^^^^^^

Similarly the output :math:`y` of parallel call is reduced using the :code:`reduce` argument.
The first dimension :math:`d` of :math:`y` is the device dimension and its name comes from the :code:`axis_name`
argument while by default is simply :code:`"device"`.

Again, let's look at a simple example::

    # This code was run on 8 simulated devices
    import objax
    import jax.numpy as jn

    net = objax.nn.Sequential([objax.nn.Linear(3, 4), objax.functional.relu])
    para_none = objax.Parallel(net, reduce=lambda x: x)
    para_first = objax.Parallel(net, reduce=lambda x: x[0])
    para_concat = objax.Parallel(net, reduce=lambda x: jn.concatenate(x))
    para_average = objax.Parallel(net, reduce=lambda x: x.mean(0))

    # A batch of mockup data
    x = objax.random.normal((96, 3))

    # We're running on multiple devices, copy the model variables to all of them first.
    with net.vars().replicate():
        print(para_none(x).shape)     # (8, 12, 4)
        print(para_first(x).shape)    # (12, 4)
        print(para_concat(x).shape)   # (96, 4)  - This is the default setting
        print(para_average(x).shape)  # (12, 4)

In the example above, the batch x (of size 96) was divided into 8 batches of size 12 by the parallel call.
Each of these batches was processed on its own device.
The final value was then reduced using the provided reduce method.

* :code:`para_none` didn't do any reduction, it just returned the value it was given, as expected is shape is
  :code:`(devices, batch // devices, ...)`.
* :code:`para_first` and :code:`para_mean` took either the first entry or the average over dimension 0, resulting in a
  shape :code:`(batch // devices, ...)`.
* :code:`para_concat` concatenated all the values resulting in a shape of :code:`(batch, ...)`.

Synchronized computations
^^^^^^^^^^^^^^^^^^^^^^^^^

So far, we only considered the case where all the devices were acting on their own, unaware of others' existence.
It's commonly desirable for devices to communicate with each other.

For example, when training a model, for efficiency one would want the optimizer to update the weights on all
the devices at the same time.
To achieve this, we would like the gradients to be computed for each sub-batch on the device, and
then **averaged across all devices**.

The good news is it is very easy to do, there are a set of predefined primitives that can be found in
:py:mod:`objax.functional.parallel` which are the direct equivalent of single device primitives:

* :py:func:`objax.functional.parallel.pmax` is the multi-device equivalent of :code:`jax.numpy.max`
* :py:func:`objax.functional.parallel.pmean` is the multi-device equivalent of :code:`jax.numpy.mean`
* and so on...

Recalling the code for the parallelized train operation::

    @objax.Function.with_vars(m.vars() + opt.vars())
    def train(x, labels, lr):
        g, v = gradient_loss(x, labels)                     # Compute gradients and loss
        opt(lr, objax.funcational.parallel.pmean(g))        # Apply averaged gradients
        return objax.funcational.parallel.pmean(v)          # Return averaged loss value

The train function is called on each device in parallel.
The :code:`objax.funcational.parallel.pmean(g)` averages the gradients :code:`g` on all devices.
Then on each device, the optimizer applies the averaged gradient to the local weight copy.
Finally the average loss is returned :code:`objax.funcational.parallel.pmean(v)`.

Vectorization
-------------

:py:class:`objax.Vectorize` is the module responsible for code vectorization.
Vectorization can be seen as a parallelization without knowledge of the devices available.
On a single GPU, vectorization parallelizes the execution in concurrent threads.
It can be combined with :code:`objax.Parallel` resulting in multi-GPU multi-threading!
Vectorization can also be done on a single CPU.
A typical example of CPU vectorization could data pre-processing or augmentation.

In its simplest form vectorization applies a function to the elements of a batch concurrently.
:py:class:`objax.Vectorize` takes a module or a function :code:`f` and vectorizes it.
Similarly to :code:`Jit` and :code:`Parallel` you must specify the variables used by the function.
Finally :code:`batch_axis` is used to say which axis should be considered as the batch axis for each input
argument of :code:`f`.
For values with no batch axis, for example when passing a value to be shared by all the calls to
the function :code:`f`, set its batch axis to :code:`None` to broadcast it.

Let's clarify this with a simple example::

    # Randomly reverse rows in a batch.
    import objax
    import jax.numpy as jn

    class RandomReverse(objax.Module):
        """Randomly reverse a single vector x and add a value y to it."""

        def __init__(self, keygen=objax.random.DEFAULT_GENERATOR):
            self.keygen = keygen

        def __call__(self, x, y):
            r = objax.random.randint([], 0, 2, generator=self.keygen)
            return x + y + r * (x[::-1] - x), r, y

    random_reverse = RandomReverse()
    vector_reverse = objax.Vectorize(random_reverse, batch_axis=(0, None))
    # vector_reverse takes two arguments (just like random_reverse), we're going to pass:
    # - a matrix x for the first argument, interpreted as a batch of vectors (batch_axis=0).
    # - a value y for the second argument, interpreted as a broadcasted value (batch_axis=None).

    # Test it on some mock up data
    x = jn.arange(20).reshape((5, 4))
    print(x)  # [[ 0  1  2  3]
              #  [ 4  5  6  7]
              #  [ 8  9 10 11]
              #  [12 13 14 15]
              #  [16 17 18 19]]

    objax.random.DEFAULT_GENERATOR.seed(1337)
    z, r, y = vector_reverse(x, 1)
    print(r)  # [0 1 0 1 1] - whether a row was reversed
    print(y)  # [1 1 1 1 1] - the brodacasted input y
    print(z)  # [[ 1  2  3  4]
              #  [ 8  7  6  5]
              #  [ 9 10 11 12]
              #  [16 15 14 13]
              #  [20 19 18 17]]

    # Above we added a single constant (y=1)
    # We can also add a vector y=(-2, -1, 0, 1)
    objax.random.DEFAULT_GENERATOR.seed(1337)
    z, r, y = vector_reverse(x, jn.array((-2, -1, 0, 1)))
    print(r)  # [0 1 0 1 1] - whether a row was reversed
    print(y)  # [[-2 -1  0  1] - the brodacasted input y
              #  [-2 -1  0  1]
              #  [-2 -1  0  1]
              #  [-2 -1  0  1]
              #  [-2 -1  0  1]]
    print(z)  # [[-2  0  2  4]
              #  [ 5  5  5  5]
              #  [ 6  8 10 12]
              #  [13 13 13 13]
              #  [17 17 17 17]]


Computing weights gradients per batch entry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a more advanced example, conceptually it is similar to what's powering differential privacy gradients::

    import objax

    m = objax.nn.Linear(3, 4)

    @objax.Function.with_vars(m.vars())
    def loss(x, y):
        return ((m(x) - y) ** 2).mean()

    g = objax.Grad(loss, m.vars())
    single_gradients = objax.Vectorize(g, batch_axis=(0, 0))  # Batch is dimension of x and y

    # Mock some data
    x = objax.random.normal((10, 3))
    y = objax.random.normal((10, 4))

    # Compute standard gradients
    print([v.shape for v in g(x, y)])              # [(4,), (3, 4)]

    # Compute per batch entry gradients
    print([v.shape for v in single_gradients(x, y)])   # [(10, 4), (10, 3, 4)]

As expected, we obtained as many gradients for each of the network's weights as there are entries in the batch.
