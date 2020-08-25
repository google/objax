Variables and Modules
=====================

Objax relies on only two concepts: Variable and Module. While not a concept in itself, another commonly used object is
the VarCollection.

* A variable is an object that has a value, typically a JaxArray.
* A module is an object that has variables and methods attached to it.
* A VarCollection is a dictionary {name: variable} with some additional methods to facilitate variable passing and
  manipulation.

The following topics are covered in this guide:

.. contents::
    :local:
    :depth: 2

Variable
--------

Variables store a value. Contrary to most frameworks variables do not require users to give them names.
Instead the **names are inferred from Python**. See :ref:`Variable names` section for more details.

Objax has four types of variables to handle the various situations encountered in machine learning:

1. :ref:`TrainVar` is a trainable variable. Its value cannot be directly modified, so as to maintain its
   differentiability.
2. :ref:`StateVar` is a state variable. It is not trainable and its value can be directly modified.
3. :ref:`TrainRef` is a reference to a TrainVar. It is a state variable used to change the value of a TrainVar, typically in the context of
   optimizers such as SGD.
4. :ref:`RandomState` is a random state variable. JAX made the design choice to require explicit random state manipulation, and this
   variable does that for you. This type of variable is used in objax.random.Generator.

.. note:: Some of variable types above take a reduce argument, this is discussed in the :ref:`Parallelism` topic.

TrainVar
^^^^^^^^

An :py:class:`objax.TrainVar` is a trainable variable.
TrainVar variables are meant to keep the trainable weights of neural networks.
As such, when calling a gradient module such as objax.GradValues, their gradients are computed.
This constrasts with the  other type of variable (state variables),
which do not have gradients.
A TrainVar is created by passing a JaxArray containing its initial value::

    import objax
    import jax.numpy as jn

    v = objax.TrainVar(jn.arange(6, dtype=jn.float32))
    print(v.value)  # [0. 1. 2. 3. 4. 5.]

    # It is not directly writable.
    v.value += 1  # Raises a ValueError

    # You can force assign it, however -as expected- all its uses before the assignment are not
    # differentiable.
    v.assign(v.value + 1)
    print(v.value)  # [1. 2. 3. 4. 5. 6.]

StateVar
^^^^^^^^

An :py:class:`objax.StateVar` is a state variable. Unlike TrainVar variables, state variables are non-trainable.
StateVar variables are used for parameters that are manually/programmatically updated.
For example, when computing a running mean of the input to a module, a StateVar is used.
StateVars are created just like TrainVars, by passing a JaxArray containing their initial value::

    import objax
    import jax.numpy as jn

    v = objax.StateVar(jn.arange(6, dtype=jn.float32))
    print(v.value)  # [0. 1. 2. 3. 4. 5.]

    # It is directly writable.
    v.value += 1
    print(v.value)  # [1. 2. 3. 4. 5. 6.]

    # You can also assign to it, it's the same as doing v.value = ...
    v.assign(v.value + 1)
    print(v.value)  # [2. 3. 4. 5. 6., 7.]

StateVar variables are ignored by gradient methods.
Unlike :ref:`TrainVar` variables, their gradients are not computed.

Why not use Python variables instead of StateVars?
""""""""""""""""""""""""""""""""""""""""""""""""""

You may be tempted to simply use Python values or numpy arrays directly since StateVars are programmatically updated.

StateVars are necessary.
They are needed to run on GPU since standard Python values and numpy arrays would not run on the GPU.
Another reason is :py:class:`objax.Jit` or :py:class:`objax.Parallel` only recognize Objax variables.

TrainRef
^^^^^^^^

An :py:class:`objax.TrainRef` is a state variable which is used to keep a reference to a :ref:`TrainVar`.
TrainRef variables are used in optimizers since optimizers need to keep a reference to the
TrainVar they are meant to optimize.
TrainRef creation differs from the previously seen variables as it takes a TrainVar as its input::

    import objax
    import jax.numpy as jn

    t = objax.TrainVar(jn.arange(6, dtype=jn.float32))
    v = objax.TrainRef(t)
    print(t.value)  # [0. 1. 2. 3. 4. 5.]

    # It is directly writable.
    v.value += 1
    print(v.value)  # [1. 2. 3. 4. 5. 6.]

    # It writes the TrainVar it references.
    print(t.value)  # [1. 2. 3. 4. 5. 6.]

    # You can also assign to it, it's the same as doing v.value = ...
    v.assign(v.value + 1)
    print(v.value)  # [2. 3. 4. 5. 6. 7.]
    print(t.value)  # [2. 3. 4. 5. 6. 7.]

TrainRef variables are ignored by gradient methods. Unlike :ref:`TrainVar` variables, their gradients are not computed.

Philosophically, one may ask why a TrainRef is needed to keep a reference to a TrainVar in an optimizer.
Indeed, why not simply keep the TrainVar itself in the optimizer?
The answer is that the optimizer is a module like any other (make sure to read the :ref:`Module` section first).
As such, one could compute the gradient of the optimizer itself.
It is for this situation that we need a TrainRef to distinguish between the optimizer's own
trainable variables (needed for its functionality) and the trainable variables of the neural network it is meant to
optimize.
It should be noted that most current optimizers do not have their own trainable variables, but we wanted to provide the
flexibility needed for future research.

RandomState
^^^^^^^^^^^

A :py:class:`objax.RandomState` is a state variable which is used to handle the tracking of random number generator
states.
It is only used in :py:class:`objax.random.Generator`.
It is responsible for automatically creating different states when the code is run in parallel in multiple GPUs
(see :py:class:`objax.Parallel`) or in a vectorized way (see :py:class:`objax.Vectorize`).
This is necessary in order for random numbers to be truly random.
In the rare event that you want to use the same random seed in a multi-GPU or vectorized module, you can use a StateVar
to store the seed.

Here's a simple example using the :py:class:`objax.random.Generator` API::

    import objax

    # Use default objax.random.DEFAULT_GENERATOR that transparently handles RandomState
    print(objax.random.normal((2,)))  # [ 0.19307713 -0.52678305]
    # A subsequent call gives, as expected new random numbers.
    print(objax.random.normal((2,)))  # [ 0.00870693 -0.04888531]

    # Make two random generators with same seeds
    g1 = objax.random.Generator(seed=1337)
    g2 = objax.random.Generator(seed=1337)

    # Random numbers using g1
    print(objax.random.normal((2,), generator=g1))  # [-0.3361883 -0.9903351]
    print(objax.random.normal((2,), generator=g1))  # [ 0.5825488 -1.4342074]

    # Random numbers using g1
    print(objax.random.normal((2,), generator=g2))  # [-0.3361883 -0.9903351]
    print(objax.random.normal((2,), generator=g2))  # [ 0.5825488 -1.4342074]
    # The result are reproducible: we obtained the same random numbers with 2 generators
    # using the same random seed.

You can also manually manipulate RandomState directly for the purpose of designing custom random numbers rules,
for example with forced correlation.
A RandomState has an extra method called :py:meth:`objax.RandomState.split` which lets it create :code:`n` new random
states.
Here's a basic example of RandomState manipulation::

    import objax

    v = objax.RandomState(1)  # 1 is the seed
    print(v.value)     # [0 1]

    # We call v.split(1) to generate 1 new state, note that split also updates v.value
    print(v.split(1))  # [[3819641963 2025898573]]
    print(v.value)     # [2441914641 1384938218]

    # We call v.split(2) to generate 2 new states, again v.value is updated
    print(v.split(2))  # [[ 622232657  209145368] [2741198523 2127103341]]
    print(v.value)     # [3514448473 2078537737]

Module
------

An :py:class:`objax.Module` is a simple container in which to store variables or other modules and on which to attach
methods that use these variables. ObJax uses the term module instead of class to avoid confusion with the Python term class.
The Module class only offers one method :py:meth:`objax.Module.vars` which returns all variables contained by the
module and its submodules in a :ref:`VarCollection`.

.. warning::
    To avoid surprising unintended behaviors, :code:`vars()` **won't look for variables or modules in lists, dicts
    or any structure** that is not a :code:`Module`.
    See [:ref:`Scope of the Module.vars method`] for how to handle lists in Objax.

Let's start with a simple example: a module called :code:`Linear`, which does a simple matrix product and adds a bias
:code:`y = x.w + b`, where :math:`w\in\mathbb{R}^{m\times n}` and :math:`b\in\mathbb{R}^n`::

    import objax
    import jax.numpy as jn

    class Linear(objax.Module):
        def __init__(self, m, n):
            self.w = objax.TrainVar(objax.random.normal((m, n)))
            self.b = objax.TrainVar(jn.zeros(n))

        def __call__(self, x):
            return x.dot(self.w.value) + self.b.value

This simple module can be used on a batch :math:`x\in\mathbb{R}^{d\times m}` to compute the resulting value
:math:`y\in\mathbb{R}^{d\times n}` for batch size :math:`d`.
Let's continue our example by creating an actual of our module and running a random batch x through it::

    f = Linear(4, 5)
    x = objax.random.normal((100, 4))  # A (100 x 4) matrix of random numbers
    y = f(x)  # y.shape == (100, 5)

We can easily make a more complicated module that uses the previously defined module Linear::

    class MiniNet(objax.Module):
        def __init__(self, m, n, p):
            self.f1 = Linear(m, n)
            self.f2 = Linear(n, p)

        def __call__(self, x):
            y = self.f1(x)
            y = objax.functional.relu(y)  # Apply a non-linearity.
            return self.f2(y)

        # You can create as many functions as you want.
        def another_function(self, x1, x2):
            return self.f2(self.f1(x1) + x2)

    f = MiniNet(4, 5, 6)
    y = f(x)  # y.shape == (100, 6)
    x2 = objax.random.normal((100, 5))  # A (100 x 5) matrix of random numbers
    another_y = f.another_function(x1, x2)

    # You can also call internal parts for example to see intermediate values.
    y1 = f.f1(x)
    y2 = objax.functional.relu(y1)
    y3 = f.f2(y2)  # y3 == y

Variable names
^^^^^^^^^^^^^^

Continuing on the previous example, let's find what the name of the variables are.
We mentioned earlier that variable names are inferred from Python and not specified by the programmer.
The way their names are inferred is from the field names, such as :code:`self.w`.
This has the benefit of ensuring consistency: a variable has a single name, and it's the name it is given in the Python
code.

Let's inspect the names::

    f = Linear(4, 5)
    print(f.vars())  # print name, size, dimensions
    # (Linear).w                 20 (4, 5)
    # (Linear).b                  5 (5,)
    # +Total(2)                  25

    f = MiniNet(4, 5, 6)
    print(f.vars())
    # (MiniNet).f1(Linear).w       20 (4, 5)
    # (MiniNet).f1(Linear).b        5 (5,)
    # (MiniNet).f2(Linear).w       30 (5, 6)
    # (MiniNet).f2(Linear).b        6 (6,)
    # +Total(4)                    61

As you can see, the names correspond to the names of the fields in which the variables are kept.

Scope of the Module.vars method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:meth:`objax.Module.vars` is meant to be simple and to remain simple.
With that in mind, we limited its scope: :code:`vars()` **won't look for variables or modules in lists, dicts or any
structure** that is not a :code:`Module`.
This is to avoid surprising unintended behavior.

Instead we made the decision to create an explicit class :py:class:`objax.ModuleList` to store a list of variables and
modules.

ModuleList
^^^^^^^^^^
The class :py:class:`objax.ModuleList` inherits from :code:`list` and behaves exactly like a list with the
difference that :code:`vars()` looks for variables and modules in it.
This class is very simple, and we invite you to look at it and use it for inspiration if you want to extend other
Python containers or design your own.

Here's a simple example of its usage::

    import objax
    import jax.numpy as jn

    class MyModule(objax.Module):
        def __init__(self):
            self.bad = [objax.TrainVar(jn.zeros(1)),
                        objax.TrainVar(jn.zeros(2))]
            self.good = objax.ModuleList([objax.TrainVar(jn.zeros(3)),
                                          objax.TrainVar(jn.zeros(4))])

    print(MyModule().vars())
    # (MyModule).good(ModuleList)[0]        3 (3,)
    # (MyModule).good(ModuleList)[1]        4 (4,)
    # +Total(2)                             7


VarCollection
-------------

The :code:`Module.vars` method returns an :py:class:`objax.VarCollection`.
This class is a dictionary that maps names to variables.
It has some additional methods and some modified behaviors specifically for variable manipulation.
In most cases, you won't need to use the more advanced methods such as :code:`__iter__`, :code:`tensors` and
:code:`assign`, but this is an in-depth topic.

Let's take a look at some of them through an example::

    import objax
    import jax.numpy as jn

    class Linear(objax.Module):
        def __init__(self, m, n):
            self.w = objax.TrainVar(objax.random.normal((m, n)))
            self.b = objax.TrainVar(jn.zeros(n))

    m1 = Linear(3, 4)
    m2 = Linear(4, 5)

    # First, as seen before, we can print the contents with print() method
    print(m1.vars())
    # (Linear).w                 12 (3, 4)
    # (Linear).b                  4 (4,)
    # +Total(2)                  16

    # A VarCollection is really a dictionary
    print(repr(m1.vars()))
    # {'(Linear).w': <objax.variable.TrainVar object at 0x7fb5e47c0ad0>,
    #  '(Linear).b': <objax.variable.TrainVar object at 0x7fb5ec017890>}

Combining multiple VarCollections is done by using addition::

    all_vars = m1.vars('m1') + m2.vars('m2')
    print(all_vars)
    # m1(Linear).w               12 (3, 4)
    # m1(Linear).b                4 (4,)
    # m2(Linear).w               20 (4, 5)
    # m2(Linear).b                5 (5,)
    # +Total(4)                  41

    # We had to specify starting names for each of the var collections since
    # they have variables with the same name. Had we not, a name collision would
    # have occured since VarCollection is a dictionary that maps names to variables.
    m1.vars() + m2.vars()  # raises ValueError('Name conflicts...')

Weight sharing
^^^^^^^^^^^^^^

It's a common technique in machine learning to share some weights.
However, it is important not to apply gradients twice or more to shared weights.
This is handled automatically by VarCollection and its :code:`__iter__` method described in the next section.
Here's a simple weight sharing example where we simply refer to the same module twice under different names::

    # Weight sharing
    shared_vars = m1.vars('m1') + m1.vars('m1_shared')
    print(shared_vars)
    # m1(Linear).w               12 (3, 4)
    # m1(Linear).b                4 (4,)
    # m1_shared(Linear).w        12 (3, 4)
    # m1_shared(Linear).b         4 (4,)
    # +Total(4)                  32


VarCollection.__iter__
^^^^^^^^^^^^^^^^^^^^^^^

Deduplication is handled automatically by the VarCollection default iterator :py:meth:`objax.VarCollection.__iter__`.
Following up on the weight sharing example above, the iterator only returns each **variable** once::

    list(shared_vars)  # [<objax.variable.TrainVar>, <objax.variable.TrainVar>]


VarCollection.tensors
^^^^^^^^^^^^^^^^^^^^^

You can collect all the values (JaxArray) for all the variables with :py:meth:`objax.VarCollection.tensors`, again in a
deduplicated manner::

    shared_vars.tensors()  # DeviceArray([[-0.1441347...]), DeviceArray([0...], dtype=float32)]

VarCollection.assign
^^^^^^^^^^^^^^^^^^^^

The last important method :py:meth:`objax.VarCollection.assign` lets you assign a tensor list to all the
VarCollection's (deduplicated) variables at once::

    shared_vars.tensors()  # DeviceArray([[-0.1441347...]), DeviceArray([0...], dtype=float32)]
    # The following increments all the variables.
    shared_vars.assign([x + 1 for x in shared_vars.tensors()])
    shared_vars.tensors()  # DeviceArray([[0.8558653...]), DeviceArray([1...], dtype=float32)]

