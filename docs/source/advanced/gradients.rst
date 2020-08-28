Understanding Gradients
=======================

Pre-requisites: :ref:`Variables and Modules`.

In this guide we discuss how to compute and use gradients in various situations and will cover the following topics:

.. contents::
    :local:
    :depth: 3

Examples will illustrate the following cases:

* Describe the :py:class:`objax.GradValues` Module.
* Show how to write your own optimizer from scratch.
* How to write a basic training iteration.
* How to handle complex gradients such as in Generative Adversarial Networks (GANs) or meta-learning.
* Explain potential optimizations in the presence of constants.

Computing gradients
-------------------

JAX, and therefore Objax, differ from most frameworks in how gradients are represented.
Gradients in JAX are represented as functions since everything in JAX is a function.
In Objax, however, they are represented as module objects.

Gradient as a module
^^^^^^^^^^^^^^^^^^^^

In machine learning, for a function :math:`f(X; \theta)`, it is common practice to separate the
inputs :math:`X` from the parameters :math:`\theta`.
Mathematically, this is captured by using a semi-colon to semantically separate one group of arguments from another.

In Objax, we represents this semantic distinction through an object :py:class:`objax.Module`:

* the module parameters :math:`\theta` are object attributes of the form :code:`self.w, ...`
* the inputs :math:`X` are arguments to the methods such as :code:`def __call__(self, x1, x2, ...):`

The gradient of a function :math:`f(X; \theta)` w.r.t to :math:`Y\subseteq X, \phi\subseteq\theta` is a function

.. math::

    g_{\scriptscriptstyle Y, \phi}(X; \theta) = (\nabla_Y f(X; \theta), \nabla_\phi f(X; \theta))

The gradient function is also a module since the same semantic distinction can be made as in `f`
between inputs :math:`X` and parameters :math:`\theta`.
Meanwhile :math:`Y, \phi` are constants of g (which inputs and which variables to compute the gradient of).
In practice, :math:`Y, \phi` are also implemented as object attributes.

The direct benefit of such a decision is that gradient manipulation is very easy and explicit: in fact it follows the
standard mathematical formulation of gradients.
While this demonstration may seem abstract, we are going to see in examples how simple it turns out to be.

A simple example
^^^^^^^^^^^^^^^^
Let's look at what gradient as a module looks like through a simple example::

    import objax

    m = objax.nn.Linear(2, 3)

    def loss(x, y):
        return ((m(x) - y) ** 2).mean()

    # Create Module that returns a tuple (g, v):
    #    g is the gradient of the loss
    #    v is the value of the loss
    gradient_loss = objax.GradValues(loss, m.vars())

    # Make up some fake data
    x = objax.random.normal((100, 2))
    y = objax.random.normal((100, 3))

    # Calling the module gradient_loss returns the actual g, v for (x, y)
    g, v = gradient_loss(x, y)
    print(v, '==', loss(x, y))  #  [DeviceArray(2.7729921, dtype=float32)] == 2.7729921
    print(g)  # A list of tensors (gradients of variables in module m)

As stated, :code:`gradient_loss` is a module instance and has variables.
Its variables are simply the ones passed to :py:class:`objax.GradValues`, we can verify it::

    print(gradient_loss.vars())
    # (GradValues)(Linear).b        3 (3,)
    # (GradValues)(Linear).w        6 (2, 3)
    # +Total(2)                     9

    # These variables are from
    print(m.vars())
    # (Linear).b                  3 (3,)
    # (Linear).w                  6 (2, 3)
    # +Total(2)                   9

Let's be clear: These are the exact same variables, not copies.
This is an instance of weight sharing, :code:`m` and :code:`gradient_loss` share the same weights.

.. _loss-optimization-label:

Loss optimization
-----------------

Gradients are useful to minimize or maximize losses.
This can be done using Stochastic Gradient Descent (SGD) with the following steps,
for a network with weights :math:`\theta` and a learning rate :math:`\mu`:

1. At iteration :math:`t`, take a batch of data :math:`x_t`
2. Compute the gradient :math:`g_t=\nabla loss(x_t)`
3. Update the weights :math:`\theta_t = \theta_{t-1} - \mu\dot g_t`
4. Goto 1

Objax already has a library of optimizers: the :ref:`objax.optimizer package`.
However we are going to create our own to demonstrate how it works with gradients.
First let's recall that everything is a Module (or a function) in Objax.
In this case, SGD will be a module, we will want to store the list of variables on which to do gradient descent.
And the function of the module will take the gradients as inputs and apply them to the variables.

Read first the part about :ref:`Variables and Modules` if you haven't done so yet. Let's get started::

    import objax

    class SGD(objax.Module):
        def __init__(self, variables: objax.VarCollection):
            self.refs = objax.ModuleList(objax.TrainRef(x)
                                         for x in variables.subset(objax.TrainVar))

        def __call__(self, lr: float, gradients: list):
            for v, g in zip(self.refs, gradients):
                v.value -= lr * g

In short, :code:`self.refs` keeps a list of references to the network trainable variables :code:`TrainVar`.
When calling the :code:`__call__` method, the values of the variables gets updated by the SGD method.

From this we can demonstrate the training of a classifier::

    import objax

    # SGD definition code from above.

    my_classifier = objax.nn.Sequential([objax.nn.Linear(2, 3), objax.functional.relu,
                                         objax.nn.Linear(3, 4)])
    opt = SGD(my_classifier.vars())

    def loss(x, labels):
        logits = my_classifier(x)
        return objax.functional.loss.cross_entropy_logits(logits, labels).mean()

    gradient_loss = objax.GradValues(loss, my_classifier.vars())

    def train(x, labels, lr):
        g, v = gradient_loss(x, labels)  # Compute gradients and loss
        opt(lr, g)                       # Apply SGD
        return v                         # Return loss value

    # Observe that the gradient contains the variables of the model (weight sharing)
    print(gradient_loss.vars())
    # (GradValues)(Sequential)[0](Linear).b        3 (3,)
    # (GradValues)(Sequential)[0](Linear).w        6 (2, 3)
    # (GradValues)(Sequential)[2](Linear).b        4 (4,)
    # (GradValues)(Sequential)[2](Linear).w       12 (3, 4)
    # +Total(4)                                   25

    # At this point you can simply call train on your training data and pass the learning rate.
    # The call will do a single step minimization the loss following the SGD method on your data.
    # Repeated calls (through various batches of data) will minimize the loss on your data.
    x = objax.random.normal((100, 2))
    labels = objax.random.randint((100,), low=0, high=4)
    labels = objax.functional.one_hot(labels, 4)
    print(train(x, labels, lr=0.01))
    # and so on...

    # See examples section for real examples.


Returning multiple values for the loss
--------------------------------------

Let's say we want to add weight decay and returning the individual components of the loss (cross-entropy, weight decay).
The loss function can return any number of values or even structures such as dicts or list.
**Only the first returned value is used to compute the gradient**, the others are returned as the loss value.

Continuing on our example, less create a new loss that returns its multiple components::

    def losses(x, labels):
        logits = my_classifier(x)
        loss_xe = objax.functional.loss.cross_entropy_logits(logits, labels).mean()
        loss_wd = sum((v.value ** 2).sum() for k, v in my_classifier.vars().items() if k.endswith('.w'))
        return loss_xe + 0.0002 * loss_wd, loss_xe, loss_wd

    gradient_losses = objax.GradValues(losses, my_classifier.vars())
    print(gradient_losses(x, labels)[1])
    # (DeviceArray(1.7454103, dtype=float32),
    #  DeviceArray(1.7434813, dtype=float32),
    #  DeviceArray(9.645493, dtype=float32))

Or one might prefer to return a dict to keep things organized::

    def loss_dict(x, labels):
        logits = my_classifier(x)
        loss_xe = objax.functional.loss.cross_entropy_logits(logits, labels).mean()
        loss_wd = sum((v.value ** 2).sum() for k, v in my_classifier.vars().items() if k.endswith('.w'))
        return loss_xe + 0.0002 * loss_wd, {'loss/xe': loss_xe, 'loss/wd': loss_wd}

    gradient_loss_dict = objax.GradValues(loss_dict, my_classifier.vars())
    print(gradient_loss_dict(x, labels)[1])
    # (DeviceArray(1.7454103, dtype=float32),
    #  {'loss/wd': DeviceArray(9.645493, dtype=float32),
    #   'loss/xe': DeviceArray(1.7434813, dtype=float32)})

Input gradients
---------------

When computing gradients it's sometimes useful to compute the gradients for some or all the inputs of the network.
For example, such functionality is needed for adversarial training or gradient penalties in GANs.
This can be easily achieved using the :code:`input_argnums` argument of :py:class:`objax.GradValues`,
here's an example::

    # Compute the gradient for my_classifier variables and for the first input of the loss:
    gradient_loss_v_x = objax.GradValues(loss, my_classifier.vars(), input_argnums=(0,))
    print(gradient_loss_v_x(x, labels)[0])
    # g = [gradient(x)] + [gradient(v) for v in classifier.vars().subset(TrainVar)]

    # Compute the gradient for my_classifier variables and for the second input of the loss:
    gradient_loss_v_y = objax.GradValues(loss, my_classifier.vars(), input_argnums=(1,))
    print(gradient_loss_v_y(x, labels)[0])
    # g = [gradient(labels)] + [gradient(v) for v in classifier.vars().subset(TrainVar)]

    # Compute the gradient for my_classifier variables and for all the inputs of the loss:
    gradient_loss_v_xy = objax.GradValues(loss, my_classifier.vars(), input_argnums=(0, 1))
    print(gradient_loss_v_xy(x, labels)[0])
    # g = [gradient(x), gradient(labels)] + [gradient(v) for v in classifier.vars().subset(TrainVar)]

    # You can also compute the gradients from the inputs alone
    gradient_loss_xy = objax.GradValues(loss, None, constants=my_classifier.vars(), input_argnums=(0, 1))
    print(gradient_loss_xy(x, labels)[0])
    # g = [gradient(x), gradient(labels)]

    # The order of the inputs matters, using input_argnums=(1, 0) instead of (0, 1)
    gradient_loss_yx = objax.GradValues(loss, None, constants=my_classifier.vars(), input_argnums=(1, 0))
    print(gradient_loss_yx(x, labels)[0])
    # g = [gradient(labels), gradient(x)]


Gradients of a subset of variables
---------------------------------

When doing more complex optimizations, one might want to temporarily treat a part of a network as constant.
This is achieved by simply passing only the variables you want the gradient of to :py:class:`objax.GradValues`.
This is useful for example in GANs where one has to optimize the discriminator and the generator
networks separately.

Continuing our example::

    all_vars = my_classifier.vars()
    print(all_vars)
    # (Sequential)[0](Linear).b        3 (3,)
    # (Sequential)[0](Linear).w        6 (2, 3)
    # (Sequential)[2](Linear).b        4 (4,)
    # (Sequential)[2](Linear).w       12 (3, 4)
    # +Total(4)                       25

Let's say we want to freeze the second Linear layer by treating it as constant::

    # We create two VarCollection
    vars_train = objax.VarCollection((k, v) for k, v in all_vars.items() if '[2](Linear)' not in k)
    print(vars_train)
    # (Sequential)[0](Linear).b        3 (3,)
    # (Sequential)[0](Linear).w        6 (2, 3)
    # +Total(2)                        9

    # We define a gradient function that ignores variables not in vars_train
    gradient_loss_freeze = objax.GradValues(loss, vars_train)
    print(gradient_loss_freeze(x, labels)[0])
    # As expected, we now have two gradient arrays, corresponding to vars_train.
    # [DeviceArray([0.19490579, 0.12267624, 0.05770121], dtype=float32),
    #  DeviceArray([[-0.21900907, -0.10813318, -0.05385721],
    #               [ 0.12701261, -0.03145855, -0.04397186]], dtype=float32)]

Higher-order gradients
----------------------

Finally one might want to optimize a loss that has a gradient in a gradient, for example let's consider the following
nested loss that relies on another loss :math:`\mathcal{L}=\texttt{loss}`:

.. math::

    \texttt{nested_loss}(x_1, y_1, x_2, y_2, \mu) = \mathcal{L}(x_1, y_1; \theta - \mu\nabla\mathcal{L}(x_2, y_2; \theta))

Implementing this in Objax remains simple, one just applies the formula verbatim.
In the following example, for the loss :math:`\mathcal{L}` we picked a cross-entropy loss but we could have picked
any other loss since :code:`nested_loss` is independent of the choice of :code:`loss`::

    train_vars = my_classifier.vars().subset(objax.TrainVar)

    def loss(x, labels):
        logits = my_classifier(x)
        return objax.functional.loss.cross_entropy_logits(logits, labels).mean()

    gradient_loss = objax.GradValues(loss, train_vars)

    def nested_loss(x1, y1, x2, y2, mu):
        # Save original network variable values
        original_values = train_vars.tensors()
        # Apply gradient from loss(x2, y2)
        for v, g in zip(train_vars, gradient_loss(x2, y2)[0]):
             v.assign(v.value - mu * g)
        # Compute loss(x1, y1)
        loss_x1y1 = loss(x1, y1)
        # Undo the gradient from loss(x2, y2)
        for v, val in zip(train_vars, original_values):
             v.assign(val)
        # Return the loss
        return loss_x1y1

    gradient_nested_loss = objax.GradValues(nested_loss, train_vars)

    # Run with mock up data, note it's only example because the loss is not for batch data.
    x1 = objax.random.normal((1, 2))
    y1 = objax.functional.one_hot(objax.random.randint((1,), low=0, high=4), 4)
    x2 = objax.random.normal((1, 2))
    y2 = objax.functional.one_hot(objax.random.randint((1,), low=0, high=4), 4)
    print(gradient_nested_loss(x1, y1, x2, y2, 0.1))
    # (gradients, loss), where the gradients are 4 tensors of the same shape as the layer variables.
    # (Sequential)[0](Linear).b        3 (3,)
    # (Sequential)[0](Linear).w        6 (2, 3)
    # (Sequential)[2](Linear).b        4 (4,)
    # (Sequential)[2](Linear).w       12 (3, 4)

Generally speaking, it is discouraged to use :py:meth:`objax.TrainVar.assign` unless you know what you are doing.
This is precisely a situation of one knowing what they are doing and it's perfectly fine to use assign here.
The reason :code:`assign` is generally discouraged is to avoid accidental bugs by overwriting a trainable variable.

On a final note, by observing that the weight update is invertible in the code above, the nested loss can be
simplified to::

    def nested_loss(x1, y1, x2, y2, mu):
        # Compute the gradient for loss(x2, y2)
        g_x2y2 = gradient_loss(x2, y2)[0]
        # Apply gradient from loss(x2, y2)
        for v, g in zip(train_vars, g_x2y2):
             v.assign(v.value - mu * g)
        # Compute loss(x1, y1)
        loss_x1y1 = loss(x1, y1)
        # Undo the gradient from loss(x2, y2)
        for v, g in zip(train_vars, g_x2y2):
             v.assign(v.value + mu * g)
        # Return the loss
        return loss_x1y1

Local gradients
^^^^^^^^^^^^^^^

In even more advanced situations, such as meta-learning, it can be desirable to have even more control over gradients.
In the above example, :code:`nested_loss` can accept vectors or matrices for its inputs :code:`x1, y1, x2, y2`.
In case of matrices, the :code:`nested_loss` is computed as:

.. math::

    \texttt{nested_loss}(X_1, Y_1, X_2, Y_2, \mu) = \mathbb{E}_{i}\mathcal{L}(X_1^{(i)}, Y_1^{(i)}; \theta - \mu\mathbb{E}_{j}\nabla\mathcal{L}(X_2^{(j)}, Y_2^{(j)}; \theta))

As a more advanced example, let's reproduce the loss from
`Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks <https://arxiv.org/abs/1703.03400>`_
in a batch form.
It is expressed as follows:

.. math::

    \texttt{nested_pairwise_loss}(X_1, Y_1, X_2, Y_2, \mu) &= \mathbb{E}_{i}\mathcal{L}(X_1^{(i)}, Y_1^{(i)}; \theta - \mu\nabla\mathcal{L}(X_2^{(i)}, Y_2^{(i)}; \theta)) \\
    &= \mathbb{E}_{i}\texttt{nested_loss}(X_1^{(i)}, Y_1^{(i)}, X_2^{(i)}, Y_2^{(i)}, \mu)

Using the previously defined :code:`nested_loss`, we can apply vectorization
(see :ref:`Vectorization` for details) on it.
In doing so we will create a module :code:`vec_nested_loss` that computes :code:`nested_loss` for all the entries
in the batches in :code:`X1, Y1, X2, Y2`::

    # Make vec_nested_loss a Module that calls nested_loss on one batch entry at a time
    vec_nested_loss = objax.Vectorize(nested_loss, gradient_loss.vars(),
                                      batch_axis=(0, 0, 0, 0, None))

    # The final loss just calls vec_nested_loss and returns the mean of the losses
    def nested_pairwise_loss(X1, Y1, X2, Y2, mu):
        return vec_nested_loss(X1, Y1, X2, Y2, mu).mean()

    # Just like any simpler loss, we can compute its gradient.
    gradient_nested_pairwise_loss = objax.GradValues(nested_pairwise_loss, vec_nested_loss.vars())

    # Run with mock up data, note it's only example because the loss is not for batch data.
    X1 = objax.random.normal((100, 2))
    Y1 = objax.functional.one_hot(objax.random.randint((100,), low=0, high=4), 4)
    X2 = objax.random.normal((100, 2))
    Y2 = objax.functional.one_hot(objax.random.randint((100,), low=0, high=4), 4)
    print(gradient_nested_pairwise_loss(X1, Y1, X2, Y2, 0.1))

Have fun!
