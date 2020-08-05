# Objax

This is not an officially supported Google product.

Objax is an object-oriented (OO) for Jax.
Objax design strives for simplicity and flexibility with the goal of facilitating experimentation and research in
machine learning.
Objax's intent is that its code should be easily understandable and forkable, as such it is particularly aimed at
students and researchers.

This the developer repository of Objax, there is very little user documentation here, for the full documentation go to
 *coming soon*.
 
You can find more information on:

* [Examples](examples/README.md)
* [Writing documentation](docs/README.md)


## User installation guide

Installing is done using pip with the following command:

```bash
pip install --upgrade objax
```

For GPU support, we assume you have already some version of CUDA installed. Here are the extra steps:

```bash
# Specify your installed CUDA version.
CUDA_VERSION=11.0
pip install --upgrade https://storage.googleapis.com/jax-releases/cuda`echo $CUDA_VERSION | sed s:\\\.::g`/jaxlib-`python3 -c 'import jaxlib; print(jaxlib.__version__)'`-`python3 -V | sed -En "s/Python ([0-9]*)\.([0-9]*).*/cp\1\2/p"`-none-manylinux2010_x86_64.whl
```

### Useful shell configurations

Here are a few useful options:

```bash
# Prevent JAX from taking the whole GPU memory
# (useful if you want to run several programs on a single GPU)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

### Testing your installation

You can run a few basic operations:

```python
import jax
import objax

print(f'Number of GPUs {jax.device_count()}')

x = objax.random.normal((100, 4))
m = objax.nn.Linear(4, 5)
print('Matrix product shape', m(x).shape)  # (100, 5)

x = objax.random.normal((100, 3, 32, 32))
m = objax.nn.Conv2D(3, 4, k=3)
print('Conv2D return shape', m(x).shape)  # (100, 4, 32, 32)
```

Typically if you get errors running this using CUDA, it probably means your installation of CUDA or CuDNN has issues.

### Installing examples

Clone the code repository:

```bash
git clone https://github.com/google/objax.git
cd objax/examples
```

## Developer installation guide

For developing purpose we recommend using virtualenv. 
Using Ubuntu or a similar Linux distribution, the setup is as follows:

```bash
# Install virtualenv if you haven't done so already
sudo apt install python3-dev python3-virtualenv python3-tk imagemagick virtualenv pandoc
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
```

It is required for the current folder to be in `PYTHONPATH`. 
This can be done with the following command:

```bash
export PYTHONPATH=$PYTHONPATH:.
```

### Useful shell configurations

Here are a few useful options:

```bash
# Prevent JAX from taking the whole GPU memory
# (useful if you want to run several programs on a single GPU)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

### Testing

```bash
./tests/run.sh
```

