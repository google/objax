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
pip install -f https://storage.googleapis.com/jax-releases/jax_releases.html jaxlib==`python3 -c 'import jaxlib; print(jaxlib.__version__)'`+cuda`echo $CUDA_VERSION | sed s:\\\.::g`
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

### Citing Objax

To cite this repository:

```
@software{objax2020github,
  author = {{Objax Developers}},
  title = {{Objax}},
  url = {https://github.com/google/objax},
  version = {1.2.0},
  year = {2020},
}
```

## Developer documentation

Here is information about
[development setup](https://objax.readthedocs.io/en/latest/dev/setup.html)
and a [guide on adding new code](https://objax.readthedocs.io/en/latest/dev/adding_module.html).
