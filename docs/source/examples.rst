Code Examples
=============

This section describes the code examples found in :code:`objax/examples`

Classification
--------------

Image
^^^^^

Example code available at :code:`examples/classify`.

:code:`examples/classify/img/logistic.py`
"""""""""""""""""""""""""""""""""""""""""

Train and evaluate a logistic regression model for binary classification on horses or humans dataset.

.. code-block:: bash

    # Run command
    python3 examples/classify/img/logistic.py

==========  =
Data        horses_or_humans from `tensorflow_datasets <https://www.tensorflow.org/datasets/api_docs/python/tfds>`_
Network     Custom single layer
Loss        :py:func:`objax.functional.loss.sigmoid_cross_entropy_logits`
Optimizer   :py:class:`objax.optimizer.SGD`
Accuracy    ~77%
Hardware    CPU or GPU or TPU
==========  =

:code:`examples/classify/img/mnist_dnn.py`
""""""""""""""""""""""""""""""""""""""""""

Train and evaluate a DNNet model for multiclass classification on the `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset.

.. code-block:: bash

    # Run command
    python3 examples/classify/img/mnist_dnn.py

==========  =
Data        `MNIST <http://yann.lecun.com/exdb/mnist/>`_ from
            `tensorflow_datasets <https://www.tensorflow.org/datasets/api_docs/python/tfds>`_
Network     Deep Neural Net :py:class:`objax.zoo.DNNet`
Loss        :py:func:`objax.functional.loss.cross_entropy_logits`
Optimizer   :py:class:`objax.optimizer.Adam`
Accuracy    ~98%
Hardware    CPU or GPU or TPU
Techniques  Model weight averaging for improved accuracy using
            :py:class:`objax.optimizer.ExponentialMovingAverage`.
==========  =

:code:`examples/classify/img/mnist_cnn.py`
""""""""""""""""""""""""""""""""""""""""""

Train and evaluate a simple custom CNN model for multiclass classification on
the `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset.

.. code-block:: bash

    # Run command
    python3 examples/classify/img/mnist_cnn.py

==========  =
Data        `MNIST <http://yann.lecun.com/exdb/mnist/>`_ from
            `tensorflow_datasets <https://www.tensorflow.org/datasets/api_docs/python/tfds>`_
Network     Custom Convolution Neural Net using :py:class:`objax.nn.Sequential`
Loss        :py:func:`objax.functional.loss.cross_entropy_logits_sparse`
Optimizer   :py:class:`objax.optimizer.Adam`
Accuracy    ~99.5%
Hardware    CPU or GPU or TPU
Techniques  * Model weight averaging for improved accuracy using
              :py:class:`objax.optimizer.ExponentialMovingAverage`.
            * Regularization using extra weight decay term in loss.
==========  =

:code:`examples/classify/img/mnist_dp.py`
"""""""""""""""""""""""""""""""""""""""""

Use differential privacy to train and evaluate a convNet model for `MNIST <http://yann.lecun.com/exdb/mnist/>`_
dataset.

.. code-block:: bash

    # Run command
    python3 examples/classify/img/mnist_dp.py
    # See available options with
    python3 examples/classify/img/mnist_dp.py --help

==========  =
Data        `MNIST <http://yann.lecun.com/exdb/mnist/>`_ from
            `tensorflow_datasets <https://www.tensorflow.org/datasets/api_docs/python/tfds>`_
Network     Custom Convolution Neural Net using :py:class:`objax.nn.Sequential`
Loss        :py:func:`objax.functional.loss.cross_entropy_logits`
Optimizer   :py:class:`objax.optimizer.SGD`
Accuracy
Hardware    GPU
Techniques  * Compute differentially private gradient using :py:class:`objax.privacy.PrivateGradValues`.
==========  =


:code:`examples/classify/img/cifar10_simple.py`
"""""""""""""""""""""""""""""""""""""""""""""""

Train and evaluate a `wide resnet <https://arxiv.org/abs/1605.07146>`_ model for multiclass classification on
the `CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ dataset.

.. code-block:: bash

    # Run command
    python3 examples/classify/img/cifar10_simple.py

==========  =
Data        `CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ from
            `tf.keras.datasets <https://www.tensorflow.org/api_docs/python/tf/keras/datasets>`_
Network     Wide ResNet using :py:class:`objax.zoo.wide_resnet.WideResNet`
Loss        :py:func:`objax.functional.loss.cross_entropy_logits_sparse`
Optimizer   :py:class:`objax.optimizer.Momentum`
Accuracy    ~91%
Hardware    GPU or TPU
Techniques  * Learning rate schedule.
            * Data augmentation (mirror / pixel shifts) in Numpy.
            * Regularization using extra weight decay term in loss.
==========  =

:code:`examples/classify/img/cifar10_advanced.py`
"""""""""""""""""""""""""""""""""""""""""""""""""

Train and evaluate convNet models for multiclass classification on
the `CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ dataset.

.. code-block:: bash

    # Run command
    python3 examples/classify/img/cifar10_advanced.py
    # Run with custom settings
    python3 examples/classify/img/cifar10_advanced.py --weight_decay=0.0001 --batch=64 --lr=0.03 --epochs=256
    # See available options with
    python3 examples/classify/img/cifar10_advanced.py --help

==========  =
Data        `CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ from
            `tensorflow_datasets <https://www.tensorflow.org/datasets/api_docs/python/tfds>`_
Network     Configurable with :code:`--arch="network"`
            * wrn28-1, wrn28-2 using :py:class:`objax.zoo.wide_resnet.WideResNet`
            * cnn32-3-max, cnn32-3-mean, cnn64-3-max, cnn64-3-mean using :py:class:`objax.zoo.convnet.ConvNet`
Loss        :py:func:`objax.functional.loss.cross_entropy_logits`
Optimizer   :py:class:`objax.optimizer.Momentum`
Accuracy    ~94%
Hardware    GPU, **Multi-GPU** or TPU
Techniques  * Model weight averaging for improved accuracy using
              :py:class:`objax.optimizer.ExponentialMovingAverage`.
            * Parallelized on multiple GPUs using :py:class:`objax.Parallel`.
            * Data augmentation (mirror / pixel shifts) in TensorFlow.
            * Cosine learning rate decay.
            * Regularization using extra weight decay term in loss.
            * Checkpointing, automatic resuming from latest checkpoint if training is interrupted using
              :py:class:`objax.io.Checkpoint`.
            * Saving of tensorboard visualization files using :py:class:`objax.jaxboard.SummaryWriter`.
            * Multi-loss reporting (cross-entropy, L2).
            * **Reusable training loop** example.
==========  =

:code:`examples/classify/img/imagenet/imagenet_train.py`
""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Train and evaluate a `ResNet50 <https://arxiv.org/abs/1603.05027>`_ model on the `ImageNet <http://www.image-net.org/>`_
dataset. See :code:`examples/classify/img/imagenet/README.md` for additional information.

==========  =
Data        `ImageNet <http://www.image-net.org/>`_ from `tensorflow_datasets <https://www.tensorflow.org/datasets/api_docs/python/tfds>`_
Network     `ResNet50 <https://arxiv.org/abs/1603.05027>`_
Loss        :py:func:`objax.functional.loss.cross_entropy_logits_sparse`
Optimizer   :py:class:`objax.optimizer.Momentum`
Accuracy
Hardware    GPU, **Multi-GPU** or TPU
Techniques  * Parallelized on multiple GPUs using :py:class:`objax.Parallel`.
            * Data augmentation (distorted bounding box crop) in TensorFlow.
            * Linear warmup followed by multi-step learning rate decay.
            * Regularization using extra weight decay term in loss.
            * Checkpointing, automatic resuming from latest checkpoint if training is interrupted using
              :py:class:`objax.io.Checkpoint`.
            * Saving of tensorboard visualization files using :py:class:`objax.jaxboard.SummaryWriter`.
==========  =

:code:`examples/classify/img/pretrained_vgg.py`
"""""""""""""""""""""""""""""""""""""""""""""""

Image classification using an ImageNet-pretrained
`VGG19 <https://www.robots.ox.ac.uk/~vgg/publications/2015/Simonyan15/simonyan15.pdf>`_ model.
See :code:`examples/classify/img/misc/PRETRAINED_VGG.md` for additional information.

==========  =
Techniques  Load VGG-19 model with pretrained weights and run 1000-way image classification.
==========  =

Semi-Supervised Learning
^^^^^^^^^^^^^^^^^^^^^^^^

Example code available at :code:`examples/semi_supervised`.

:code:`examples/semi_supervised/img/fixmatch.py`
""""""""""""""""""""""""""""""""""""""""""""""""

Semi-supervised learning of image classification models with `FixMatch <https://arxiv.org/abs/2001.07685>`_.

.. code-block:: bash

    # Run command
    python3 examples/classify/semi_supervised/img/fixmatch.py
    # Run with custom settings
    python3 examples/classify/semi_supervised/img/fixmatch.py --dataset=cifar10.3@1000-0
    # See available options with
    python3 examples/classify/semi_supervised/img/fixmatch.py --help

==========  =
Data        `CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_, `CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_, `SVHN <http://ufldl.stanford.edu/housenumbers/>`_, `STL10 <https://ai.stanford.edu/~acoates/stl10/>`_
Network     Custom implementation of Wide ResNet.
Loss        :py:func:`objax.functional.loss.cross_entropy_logits` and :py:func:`objax.functional.loss.cross_entropy_logits_sparse`
Optimizer   :py:class:`objax.optimizer.Momentum`
Accuracy    See `paper <https://arxiv.org/abs/2001.07685>`_
Hardware    GPU, **Multi-GPU**, TPU
Techniques  * Load data from multiple data pipelines.
            * Advanced data augmentation such as `RandAugment <https://arxiv.org/abs/1909.13719>`_ and
              `CTAugment <https://arxiv.org/abs/1911.09785>`_.
            * Stop gradient using :py:func:`objax.functional.stop_gradient`.
            * Cosine learning rate decay.
            * Regularization using extra weight decay term in loss.
==========  =

GPT-2
-----

Example code available at :code:`examples/gpt-2`.

:code:`examples/gpt-2/gpt2.py`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Load pretrained `GPT2 <https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf>`_
model (124M parameter) and demonstrate how to use the model to generate a text sequence.
See :code:`examples/gpt-2/README.md` for additional information.

==========  =
Hardware    GPU or TPU
Techniques  * Define Transformer model.
            * Load GPT2 model with pretrained weights and generate a sequence.
==========  =

RNN
---

Example code is available at :code:`examples/rnn`.

:code:`examples/rnn/shakespeare.py`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Train and evaluate a vanilla RNN model on the Shakespeare corpus dataset.
See :code:`examples/rnn/README.md` for additional information.

.. code-block:: bash

    # Run command
    python3 examples/rnn/shakespeare.py

==========  =
Data        `Shakespeare corpus <https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt>`_
            from `tensorflow_datasets <https://www.tensorflow.org/datasets/api_docs/python/tfds>`_
Network     Custom implementation of vanilla RNN.
Loss        :py:func:`objax.functional.loss.cross_entropy_logits`
Optimizer   :py:class:`objax.optimizer.Adam`
Hardware    GPU or TPU
Techniques  * Model weight averaging for improved accuracy using :py:class:`objax.optimizer.ExponentialMovingAverage`.
            * Data pipeline of sequence data for training.
            * Data processing (e.g., tokenize).
            * Clip gradients.
==========  =


Optimization
------------

Example codes available at :code:`examples/optimization`.

:code:`examples/optimization/maml.py`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Meta-learning method `MAML <https://arxiv.org/abs/1703.03400>`_ implementation to demonstrate computing the gradient of 
a gradient.

.. code-block:: bash

    # Run command
    python3 examples/optimization/maml.py

==========  =
Data        Synthetic data
Network     3-layer DNNet
Hardware    CPU or GPU or TPU
Techniques  Gradient of gradient.
==========  =

Jaxboard
--------

Example code available at :code:`examples/jaxboard`.

:code:`examples/jaxboard/summary.py`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sample usage of jaxboard.

.. code-block:: bash

    # Run command
    python3 examples/jaxboard/summary.py

==========  =
Hardware    CPU
Usages      * summary scalar
            * summary text
            * summary image
==========  =


