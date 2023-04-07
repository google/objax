objax package
=============

.. currentmodule:: objax

.. contents::
    :local:
    :depth: 1

Modules
-------

.. autosummary::

    Module
    ModuleList
    ForceArgs
    Function
    Grad
    GradValues
    Jacobian
    Hessian
    Jit
    Parallel
    Vectorize

.. autoclass:: Module
   :members:

.. autoclass:: ModuleList
    :members:
    :show-inheritance:

    .. seealso:: :py:class:`objax.nn.Sequential`

    Usage example::

        import objax

        ml = objax.ModuleList(['hello', objax.TrainVar(objax.random.normal((10,2)))])
        print(ml.vars())
        # (ModuleList)[1]            20 (10, 2)
        # +Total(1)                  20

        ml.pop()
        ml.append(objax.nn.Linear(2, 3))
        print(ml.vars())
        # (ModuleList)[1](Linear).b        3 (3,)
        # (ModuleList)[1](Linear).w        6 (2, 3)
        # +Total(2)                        9


.. autoclass:: Function
    :members: vars

    Usage example::

        import objax
        import jax.numpy as jn

        m = objax.nn.Linear(2, 3)

        def f1(x, y):
            return ((m(x) - y) ** 2).mean()

        # Method 1: Create module by calling objax.Function to tell which variables are used.
        m1 = objax.Function(f1, m.vars())

        # Method 2: Use the decorator.
        @objax.Function.with_vars(m.vars())
        def f2(x, y):
            return ((m(x) - y) ** 2).mean()

        # All behave like functions
        x = jn.arange(10).reshape((5, 2))
        y = jn.arange(15).reshape((5, 3))
        print(type(f1), f1(x, y))  # <class 'function'> 237.01947
        print(type(m1), m1(x, y))  # <class 'objax.module.Function'> 237.01947
        print(type(f2), f2(x, y))  # <class 'objax.module.Function'> 237.01947

    Usage of `Function` is not necessary: it is made available for aesthetic reasons (to accomodate for users personal
    taste). It is also used internally to keep the code simple for Grad, Jit, Parallel, Vectorize and future primitives.

    .. automethod:: with_vars

        Usage example::

            import objax

            m = objax.nn.Linear(2, 3)

            @objax.Function.with_vars(m.vars())
            def f(x, y):
                return ((m(x) - y) ** 2).mean()

            print(type(f))  # <class 'objax.module.Function'>

    .. automethod:: auto_vars

        Usage example::

            import objax

            m = objax.nn.Linear(2, 3)

            @objax.Function.auto_vars
            def f(x, y):
                return ((m(x) - y) ** 2).mean()

            print(type(f))  # <class 'objax.module.Function'>

.. autoclass:: ForceArgs
    :members:

    One example of `ForceArgs` usage is to override `training` argument for batch normalization::

        import objax
        from objax.zoo.resnet_v2 import ResNet50

        model = ResNet50(in_channels=3, num_classes=1000)

        # Modify model to force training=False on first resnet block.
        # First two ops in the resnet are convolution and padding,
        # resnet blocks are starting at index 2.
        model[2] = objax.ForceArgs(model[2], training=False)

        # model(x, training=True) will be using `training=False` on model[2] due to ForceArgs
        # ...

        # Undo specific value of forced arguments in `model` and all submodules of `model`
        objax.ForceArgs.undo(model, training=True)

        # Undo all values of specific argument in `model` and all submodules of `model`
        objax.ForceArgs.undo(model, training=objax.ForceArgs.ANY)

        # Undo all values of all arguments in `model` and all submodules of `model`
        objax.ForceArgs.undo(model)


.. autoclass:: Grad
   :members:

    Usage example::

        import objax

        m = objax.nn.Sequential([objax.nn.Linear(2, 3), objax.functional.relu])

        def f(x, y):
            return ((m(x) - y) ** 2).mean()

        # Create module to compute gradients of f for m.vars()
        grad_f = objax.Grad(f, m.vars())

        # Create module to compute gradients of f for input 0 (x) and m.vars()
        grad_fx = objax.Grad(f, m.vars(), input_argnums=(0,))


    For more information and examples, see :ref:`Understanding Gradients`.

.. autoclass:: GradValues
   :members:

    Usage example::

        import objax

        m = objax.nn.Sequential([objax.nn.Linear(2, 3), objax.functional.relu, objax.nn.Linear(3, 2)])

        @objax.Function.with_vars(m.vars())
        def f(x, y):
            return ((m(x) - y) ** 2).mean()

        # Create module to compute gradients of f for m.vars()
        grad_val_f = objax.GradValues(f, m.vars())

        # Create module to compute gradients of f for only some variables
        grad_val_f_head = objax.GradValues(f, m[:1].vars())

        # Create module to compute gradients of f for input 0 (x) and m.vars()
        grad_val_fx = objax.GradValues(f, m.vars(), input_argnums=(0,))


    For more information and examples, see :ref:`Understanding Gradients`.

.. autoclass:: Jacobian
   :members:

    Usage example::

        import objax
        import jax.numpy as jn

        data = jn.array([1.0, 2.0, 3.0, 4.0])

        w = objax.TrainVar(jn.array([[1., 2., 3., 4.],
                                     [5., 6., 7., 8.],
                                     [9., 0., 1., 2.]]))
        b = objax.TrainVar(jn.array([-1., 0., 1.]))

        # f_lin(x) = w*x + b
        @objax.Function.with_vars(objax.VarCollection({'w': w, 'b': b}))
        def f_lin(x):
            return jn.dot(W.value, x) + b.value

        # Jacobian w.r.t. model variables
        jac_vars_module = objax.Jacobian(f_lin, f_lin.vars())
        j = jac_vars_module(data)

        # Jacobian w.r.t. arguments
        jac_x_module = objax.Jacobian(f_lin, None, input_argnums=(0,))
        j = jac_x_module(data)


.. autoclass:: Hessian
   :members:

    Usage example::

        import objax
        import jax.numpy as jn

        data = jn.array([1.0, 2.0, 3.0, 4.0])

        w = objax.TrainVar(jn.array([[1., 2., 3., 4.],
                                     [5., 6., 7., 8.],
                                     [9., 0., 1., 2.]]))
        b = objax.TrainVar(jn.array([-1., 0., 1.]))

        # f_sq(x) = (w*x + b)^2
        @objax.Function.with_vars(objax.VarCollection({'w': w, 'b': b}))
        def f_sq(x):
            h = jn.dot(w.value, x) + b.value
            return jn.dot(h, h)

        # Hessian w.r.t. both variables and input argument
        hess_module = objax.Hessian(f_sq, f_sq.vars(), input_argnums=(0,))
        h = hess_module(data)


.. autoclass:: Jit
    :members: vars

    Usage example::

        import objax

        m = objax.nn.Sequential([objax.nn.Linear(2, 3), objax.functional.relu])
        jit_m = objax.Jit(m)                          # Jit a module

        # For jitting functions, use objax.Function.with_vars
        @objax.Function.with_vars(m.vars())
        def f(x):
            return m(x) + 1

        jit_f = objax.Jit(f)

    For more information, refer to :ref:`JIT Compilation`.
    Also note that one can pass variables to be used by Jit for a module `m`: the rest will be optimized away as
    constants, for more information refer to :ref:`Constant optimization`.

.. autoclass:: Parallel
   :members: vars

    Usage example::

        import objax

        m = objax.nn.Sequential([objax.nn.Linear(2, 3), objax.functional.relu])
        para_m = objax.Parallel(m)                         # Parallelize a module

        # For parallelizing functions, use objax.Function.with_vars
        @objax.Function.with_vars(m.vars())
        def f(x):
            return m(x) + 1

        para_f = objax.Parallel(f)

    When calling a parallelized module, one must replicate the variables it uses on all devices::

        x = objax.random.normal((16, 2))
        with m.vars().replicate():
            y = para_m(x)

    For more information, refer to :ref:`Parallelism`.
    Also note that one can pass variables to be used by Parallel for a module `m`: the rest will be optimized away as
    constants, for more information refer to :ref:`Constant optimization`.

.. autoclass:: Vectorize
   :members: vars

    Usage example::

        import objax

        m = objax.nn.Sequential([objax.nn.Linear(2, 3), objax.functional.relu])
        vec_m = objax.Vectorize(m)                         # Vectorize a module

        # For vectorizing functions, use objax.Function.with_vars
        @objax.Function.with_vars(m.vars())
        def f(x):
            return m(x) + 1

        vec_f = objax.Parallel(f)

    For more information and examples, refer to :ref:`Vectorization`.

Variables
---------

.. autosummary::

    BaseVar
    TrainVar
    BaseState
    StateVar
    TrainRef
    RandomState
    VarCollection

.. autoclass:: BaseVar

.. autoclass:: TrainVar
   :members:
   :inherited-members:

.. autoclass:: BaseState

.. autoclass:: TrainRef
   :members:
   :inherited-members:

.. autoclass:: StateVar
   :members:
   :inherited-members:

.. autoclass:: RandomState
   :members:
   :inherited-members:

.. autoclass:: VarCollection

    Usage example::

        import objax

        m = objax.nn.Sequential([objax.nn.Linear(2, 3), objax.functional.relu])
        vc = m.vars()  # This is a VarCollection

        # It is a dictionary
        print(repr(vc))
        # {'(Sequential)[0](Linear).b': <objax.variable.TrainVar object at 0x7faecb506390>,
        #  '(Sequential)[0](Linear).w': <objax.variable.TrainVar object at 0x7faec81ee350>}
        print(vc.keys())  # dict_keys(['(Sequential)[0](Linear).b', '(Sequential)[0](Linear).w'])
        assert (vc['(Sequential)[0](Linear).w'].value == m[0].w.value).all()

        # Convenience print
        print(vc)
        # (Sequential)[0](Linear).b        3 (3,)
        # (Sequential)[0](Linear).w        6 (2, 3)
        # +Total(2)                        9

        # Extra methods for manipulation of variables:
        # For example, increment all variables by 1
        vc.assign([x+1 for x in vc.tensors()])

        # It's used by other modules.
        # For example it's used to tell what variables are used by a function.

        @objax.Function.with_vars(vc)
        def my_function(x):
            return objax.functional.softmax(m(x))

    For more information and examples, refer to :ref:`VarCollection`.

    .. automethod:: assign
    .. automethod:: rename

        Renaming entries in a `VarCollection` is a powerful tool that can be used for

        - mapping weights between models that differ slightly.
        - loading data checkpoints from foreign ML frameworks.

        Usage example::

            import re
            import objax

            m = objax.nn.Sequential([objax.nn.Linear(2, 3), objax.functional.relu])
            print(m.vars())
            # (Sequential)[0](Linear).b        3 (3,)
            # (Sequential)[0](Linear).w        6 (2, 3)
            # +Total(2)                        9

            # For example remove modules from the name
            renamer = objax.util.Renamer([(re.compile('\([^)]+\)'), '')])
            print(m.vars().rename(renamer))
            # [0].b                       3 (3,)
            # [0].w                       6 (2, 3)
            # +Total(2)                   9

            # One can chain renamers, their syntax is flexible and it can use a string mapping:
            renamer_all = objax.util.Renamer({'[': '.', ']': ''}, renamer)
            print(m.vars().rename(renamer_all))
            # .0.b                        3 (3,)
            # .0.w                        6 (2, 3)
            # +Total(2)                   9


    .. automethod:: replicate
    .. automethod:: subset
    .. automethod:: tensors
    .. automethod:: update


Constants
---------

.. autoclass:: ConvPadding
   :members:
