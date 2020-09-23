# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = ['EasyDict', 'args_indexes', 'dummy_context_mgr', 'ilog2', 'local_kwargs', 'map_to_device',
           'multi_host_barrier', 'positional_args_names', 'to_tuple']

import contextlib
import inspect
from numbers import Number
from typing import Callable, List, Union, Tuple, Iterable

import jax
import jax.numpy as jn
import numpy as np
from jax.interpreters.pxla import ShardedDeviceArray


class EasyDict(dict):
    """Custom dictionary that allows to access dict values as attributes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def args_indexes(f: Callable, args: Iterable[str]) -> Iterable[int]:
    """Returns the indexes of variable names of a function."""
    d = {name: i for i, name in enumerate(positional_args_names(f))}
    for name in args:
        index = d.get(name)
        if index is None:
            raise ValueError(f'Function {f} does not have argument of name {name}', (f, name))
        yield index


@contextlib.contextmanager
def dummy_context_mgr():
    """Empty Context Manager."""
    yield None


def ilog2(x: float):
    """Integer log2."""
    return int(np.ceil(np.log2(x)))


def local_kwargs(kwargs: dict, f: Callable) -> dict:
    """Return the kwargs from dict that are inputs to function f."""
    s = inspect.signature(f)
    p = s.parameters
    if next(reversed(p.values())).kind == inspect.Parameter.VAR_KEYWORD:
        return kwargs
    if len(kwargs) < len(p):
        return {k: v for k, v in kwargs.items() if k in p}
    return {k: kwargs[k] for k in p.keys() if k in kwargs}


map_to_device: Callable[[List[jn.ndarray]], List[ShardedDeviceArray]] = jax.pmap(lambda x: x, axis_name='device')


def multi_host_barrier():
    """Barrier op for multi-host setup."""
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()


def positional_args_names(f: Callable) -> List[str]:
    """Returns the ordered names of the positional arguments of a function."""
    return list(p.name for p in inspect.signature(f).parameters.values()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD))


def to_tuple(v: Union[Tuple[Number, ...], Number, Iterable], n: int):
    """Converts input to tuple."""
    if isinstance(v, tuple):
        return v
    elif isinstance(v, Number):
        return (v,) * n
    else:
        return tuple(v)
