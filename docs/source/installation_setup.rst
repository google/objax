Installation and Setup
======================

For developing or contributing to Objax, see :ref:`Development setup`.

User installation
-----------------

Install using :code:`pip` with the following command:

.. code-block:: bash

    pip install --upgrade objax

For GPU support, we assume you have already some version of CUDA installed. Here are the extra steps:

.. code-block:: bash

    # Specify your installed CUDA version.
    CUDA_VERSION=11.0
    pip install -f https://storage.googleapis.com/jax-releases/jax_releases.html jaxlib==`python3 -c 'import jaxlib; print(jaxlib.__version__)'`+cuda`echo $CUDA_VERSION | sed s:\\\.::g`

Useful shell configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here are a few useful options:

.. code-block:: bash

    # Prevent JAX from taking the whole GPU memory
    # (useful if you want to run several programs on a single GPU)
    export XLA_PYTHON_CLIENT_PREALLOCATE=false

Testing your installation
^^^^^^^^^^^^^^^^^^^^^^^^^

You can run the code below to test your installation::

    import jax
    import objax

    print(f'Number of GPUs {jax.device_count()}')

    x = objax.random.normal((100, 4))
    m = objax.nn.Linear(4, 5)
    print('Matrix product shape', m(x).shape)  # (100, 5)

    x = objax.random.normal((100, 3, 32, 32))
    m = objax.nn.Conv2D(3, 4, k=3)
    print('Conv2D return shape', m(x).shape)  # (100, 4, 32, 32)

If you get errors running this using CUDA, it probably means your installation of CUDA or CuDNN has issues.

Installing examples
^^^^^^^^^^^^^^^^^^^

Clone the code repository:

.. code-block:: bash

    git clone https://github.com/google/objax.git
    cd objax/examples
