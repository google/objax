# Objax

[**Tutorials**](https://objax.readthedocs.io/en/latest/notebooks/Objax_Basics.html)
| [**Install**](https://objax.readthedocs.io/en/latest/installation_setup.html)
| [**Documentation**](https://objax.readthedocs.io/en/latest/)
| [**Philosophy**](https://objax.readthedocs.io/en/latest/index.html#objax-philosophy)

This is not an officially supported Google product.

Objax is an open source machine learning framework that accelerates research and learning thanks to a
minimalist object-oriented design and a readable code base.
Its name comes from the contraction of Object and [JAX](https://github.com/google/jax) -- a popular high-performance
framework.
Objax is designed **by researchers for researchers** with a focus on simplicity and understandability.
Its users should be able to easily read, understand, extend, and modify it to fit their needs.

This is the developer repository of Objax, there is very little user documentation
here, for the full documentation go to [objax.readthedocs.io](https://objax.readthedocs.io/).

You can find READMEs in the subdirectory of this project, for example:

* [Sample Code](examples/README.md)
* [Writing documentation](docs/README.md)


## User installation guide

You install Objax using `pip` as follows:

```bash
pip install --upgrade objax
```

Objax supports GPUs but assumes that you already have some version of CUDA
installed. Here are the extra steps:

```bash
# Update accordingly to your installed CUDA version
CUDA_VERSION=11.0
pip install --upgrade https://storage.googleapis.com/jax-releases/cuda`echo $CUDA_VERSION | sed s:\\\.::g`/jaxlib-`python3 -c 'import jaxlib; print(jaxlib.__version__)'`+cuda`echo $CUDA_VERSION | sed s:\\\.::g`-`python3 -V | sed -En "s/Python ([0-9]*)\.([0-9]*).*/cp\1\2/p"`-none-manylinux2010_x86_64.whl
```

### Useful environment configurations

Here are a few useful options:

```bash
# Prevent JAX from taking the whole GPU memory
# (useful if you want to run several programs on a single GPU)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

### Testing your installation

You can test your installation by running the code below:

```python
import jax
import objax

print(f'Number of GPUs {jax.device_count()}')

x = objax.random.normal(shape=(100, 4))
m = objax.nn.Linear(nin=4, nout=5)
print('Matrix product shape', m(x).shape)  # (100, 5)

x = objax.random.normal(shape=(100, 3, 32, 32))
m = objax.nn.Conv2D(nin=3, nout=4, k=3)
print('Conv2D return shape', m(x).shape)  # (100, 4, 32, 32)
```

Typically if you get errors running this using CUDA, it probably means your
installation of CUDA or CuDNN has issues.

### Runing code examples

Clone the code repository:

```bash
git clone https://github.com/google/objax.git
cd objax/examples
```

## Developer installation guide

We recommend using `virtualenv` if you want to develop in Objax. The setup for
Ubuntu or a similar Linux distribution is as follows:

```bash
# Install virtualenv if you haven't done so already
sudo apt install python3-dev python3-virtualenv python3-tk imagemagick virtualenv pandoc
# Create a virtual environment (for example ~/jax3, you can use your name here)
virtualenv -p python3 --system-site-packages ~/jax3
# Start the virtual environment
. ~/jax3/bin/activate

# Clone objax git repository, if you haven't.
git clone https://github.com/google/objax.git
cd objax

# Install python dependencies.
pip install --upgrade -r requirements.txt
pip install --upgrade -r docs/requirements.txt
pip install --upgrade -r examples/requirements.txt

# If you have CUDA installed, specify your installed CUDA version.
CUDA_VERSION=11.0
pip install --upgrade https://storage.googleapis.com/jax-releases/cuda`echo $CUDA_VERSION | sed s:\\\.::g`/jaxlib-`python3 -c 'import jaxlib; print(jaxlib.__version__)'`+cuda`echo $CUDA_VERSION | sed s:\\\.::g`-`python3 -V | sed -En "s/Python ([0-9]*)\.([0-9]*).*/cp\1\2/p"`-none-manylinux2010_x86_64.whl
```

The current folder must be in `PYTHONPATH`. You can do this with the following command:

```bash
export PYTHONPATH=$PYTHONPATH:.
```

### Running linter and tests

Install additional packages for testing and linting:

```bash
# Installation of pytest is optional.
# Tests will run without it, however pytest provides nicer output and
# GitHub tests are run using pytest.
pip install pytest

# Flake8 is required to run linter.
pip install flake8
```

Run linter:

```bash
./tests/run_linter.sh
```

Run tests:

```bash
./tests/run_tests.sh
```
