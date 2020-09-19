Welcome to Objax's documentation!
=================================

Objax is an open source machine learning framework that accelerates research and learning thanks to a
minimalist object-oriented design and a readable code base.
Its name comes from the contraction of Object and `JAX <https://github.com/google/jax>`_ -- a popular high-performance
framework.
Objax is designed **by researchers for researchers** with a focus on simplicity and understandability.
Its users should be able to easily read, understand, extend, and modify it to fit their needs.

:doc:`Try the 5 minutes tutorial. <notebooks/Objax_Basics>`

Machine learning's :code:`'Hello world'`: optimizing the weights of classifier ``net`` through gradient descent::

   opt = objax.optimizer.Adam(net.vars())

   def loss(x, y):
       logits = net(x)  # Output of classifier on x
       xe = cross_entropy_logits(logits, y)
       return xe.mean()

   # Perform gradient descent wrt to net weights    
   gv = objax.GradValues(loss, net.vars())

   def train_op(x, y):
       g, v = gv(x, y)  # returns gradients g and loss v
       opt(lr, g)  # update weights
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
Often they read the code of some technique, say an Adam optimizer, to understand how it works
so they can extend it or design a new optimizer.
This is how machine learning frameworks differ from standard libraries: a large class of
users not only look at the APIs but also at the code behind these APIs.

Coded for simplicity
^^^^^^^^^^^^^^^^^^^^

Source code should be understandable by everyone, including users without background in computer science.
So how simple is it really? Judge for yourself with this tutorial: :doc:`notebooks/Logistic_Regression`.

Object-oriented
^^^^^^^^^^^^^^^
It is common in machine learning to separate the inputs (:math:`X`)
from the parameters (:math:`\theta`) of a function :math:`f(X;
\theta)`.
Math notation captures this difference by using a semi-colon to semantically separate the first group of arguments from the other.  

Objax represents this semantic distinction through :py:class:`objax.Module`:

* the module's parameters :math:`\theta` are attributes of the form :code:`self.w, ...`
* inputs :math:`X` are method arguments such as :code:`def __call__(self, x, y, ...):`

Designed for flexibility
^^^^^^^^^^^^^^^^^^^^^^^^

Objax minimizes the number of abstractions users need to understand. There are two main ones: *Modules* and *Variables*.
Everything is built out of these two basic classes. You can read more about this in :doc:`advanced/variables_and_modules`. 

Engineered for performance
^^^^^^^^^^^^^^^^^^^^^^^^^^

In machine learning, performance is essential.
Every second counts.
Objax makes it count by using the JAX/XLA engine that also powers TensorFlow.
Read more about this in :doc:`advanced/jit`.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation_setup
   notebooks/Objax_Basics
   notebooks/Logistic_Regression
   notebooks/Custom_Networks
   examples
   tutorials

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
