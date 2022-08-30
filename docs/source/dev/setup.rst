Development setup
=================

This section describes some basic setup to start developing and extending Objax.

Environment setup
-----------------

First of all you need to install all necessary dependencies.
We recommend to setup a separate :code:`virtualenv` to work on Objax,
it could be done with following commands on Ubuntu or similar Linux distribution:

.. code-block:: bash

    # Install virtualenv if you haven't done so already
    sudo apt install python3-dev python3-virtualenv python3-tk imagemagick virtualenv pandoc
    # Create a virtual environment (for example ~/.venv/objax, you can use your name here)
    virtualenv -p python3 --system-site-packages ~/.venv/objax
    # Start the virtual environment
    . ~/.venv/objax/bin/activate

    # Clone objax git repository, if you haven't.
    git clone https://github.com/google/objax.git
    cd objax

    # Install python dependencies.
    pip install --upgrade -r requirements.txt
    pip install --upgrade -r tests/requirements.txt
    pip install --upgrade -r docs/requirements.txt
    pip install --upgrade -r examples/requirements.txt
    pip install flake8

    # jaxlib releases require CUDA 11.2 or newer
    RELEASE_URL="https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
    JAX_VERSION=`python3 -c 'import jax; print(jax.__version__)'`
    pip uninstall -y jaxlib
    pip install -f $RELEASE_URL jax[cuda]==$JAX_VERSION

Running tests and linter
------------------------

Run linter:

.. code-block:: bash

    ./tests/run_linter.sh

Run tests:

.. code-block:: bash

    ./tests/run_tests.sh

Running a single test:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES= python3 -m unittest tests/jit.py
