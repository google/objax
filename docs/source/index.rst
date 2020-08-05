Welcome to Objax's documentation!
=================================

**Objax** adds an **Object Oriented** layer on top of `JAX <https://github.com/google/jax>`_
(a framework for high performance machine learning).
Objax is designed **by researchers for researchers**, focusing on simplicity and understandability.
The library aims to make it easy for its users to read, understand, extend, and modify its code to adapt it
to their needs.

:doc:`Try the 5 minutes tutorial. <notebooks/Objax_Basics>`

Machine learning's :code:`'Hello world'`: A classifier's weigths optimization with gradient descent::

   opt = objax.optimizer.Adam(net.vars())

   def loss(x, y):
       logits = net(x)
       xe = cross_entropy_logits(logits, y)
       return xe.mean()

   gv = objax.GradValues(loss, net.vars())

   def train_op(x, y):
       g, v = gv(x, y)
       opt(lr, g)
       return v

   train_op = objax.Jit(train_op, net.vars() + opt.vars())

Objax philosophy
----------------

.. epigraph::

   Objax pursues the quest for the **simplest design and code** that's as **easy** as possible **to extend**
   without sacrificing **performance**.

    -- Objax Devs

Motivation
^^^^^^^^^^

Researchers and students look at machine learning frameworks in their own way.
Often they find themselves trying to read the code of some technique, say an Adam optimizer, to get an idea
how it works or to try to extend it or as the basis of research for a new optimizer.
Case in point is machine learning frameworks differ from standard libraries in how they are used: a large class of
users not only look at the APIs but also at the code behind these APIs.

Coded for simplicity
^^^^^^^^^^^^^^^^^^^^

Source code should be understandable by everyone, including from users without backgrounds in computer science.
So how simple is it really? Judge for yourself with this tutorial: :doc:`notebooks/Logistic_Regression`.

Object-oriented
^^^^^^^^^^^^^^^
In machine learning, for a function :math:`f(X; \theta)`, it is common practice to separate the
inputs :math:`X` from the parameters :math:`\theta`.
Mathematically, this is captured by using a semi-colon to semantically separate one group of arguments from another.

In Objax, we represents this semantic distinction through an object :py:class:`objax.Module`:

* the module parameters :math:`\theta` are object attributes of the form :code:`self.w, ...`
* the inputs :math:`X` are arguments to the methods such as :code:`def __call__(self, x, y, ...):`

Designed for flexibility
^^^^^^^^^^^^^^^^^^^^^^^^

We minimized the number of abstractions, in fact there are only two main ones: the Module and the Variable.
Everything is built out of these two basic classes. Read more about this in this in-depth guide
:doc:`advanced/variables_and_modules`.

Engineered for performance
^^^^^^^^^^^^^^^^^^^^^^^^^^

In machine learning, performance is essential.
Every second counts.
And with Objax we make it count by using JAX/XLA engine which also powers TensorFlow.
Read more about this in :doc:`advanced/jit`.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation_setup
   notebooks/Objax_Basics
   notebooks/Logistic_Regression
   notebooks/Custom_Networks
   examples

.. toctree::
   :maxdepth: 2
   :caption: API documentation

   objax/index

.. toctree::
   :maxdepth: 2
   :caption: In-depth topics

   advanced/variables_and_modules
   advanced/gradients
   advanced/jit
   advanced/io

  

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
