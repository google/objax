Installation and Setup
======================

User installation
-----------------

Install using :code:`pip` with the following command:

.. code-block:: bash

    pip install --upgrade objax

For GPU support, we assume you have already some version of CUDA installed. Here are the extra steps:  

.. code-block:: bash

    # Specify your installed CUDA version.
    CUDA_VERSION=11.0
    pip install --upgrade https://storage.googleapis.com/jax-releases/cuda`echo $CUDA_VERSION | sed s:\\\.::g`/jaxlib-`python3 -c 'import jaxlib; print(jaxlib.__version__)'`-`python3 -V | sed -En "s/Python ([0-9]*)\.([0-9]*).*/cp\1\2/p"`-none-manylinux2010_x86_64.whl

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


Developer installation
----------------------

For developing purpose we recommend using :code:`virtualenv`.
The setup in Ubuntu or similar Linux distributions is as follows:

.. code-block:: bash

    # Install virtualenv if you haven't done so already
    sudo apt install python3-dev python3-virtualenv python3-tk imagemagick virtualenv
    # Create a virtual environment (for example ~/jax3, you can use your name here)
    virtualenv -p python3 --system-site-packages ~/jax3
    # Start the virtual environment
    . ~/jax3/bin/activate

    # Clone objax git repository.
    git clone https://github.com/google/objax.git
    cd objax

    # Install python dependencies.
    pip install --upgrade -r requirements.txt
    pip install --upgrade -r docs/requirements.txt
    pip install --upgrade -r examples/requirements.txt

    # If you have CUDA installed, specify your installed CUDA version.
    CUDA_VERSION=11.0
    pip install --upgrade https://storage.googleapis.com/jax-releases/cuda`echo $CUDA_VERSION | sed s:\\\.::g`/jaxlib-`python3 -c 'import jaxlib; print(jaxlib.__version__)'`-`python3 -V | sed -En "s/Python ([0-9]*)\.([0-9]*).*/cp\1\2/p"`-none-manylinux2010_x86_64.whl

The current folder must be in :code:`PYTHONPATH`.
This can be done with the following command: 

.. code-block:: bash

    export PYTHONPATH=$PYTHONPATH:.

.. seealso:: :ref:`Useful shell configurations`
