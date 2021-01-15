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


__all__ = ['EasyDict', 'args_indexes', 'class_name', 'dummy_context_mgr', 'ilog2', 'local_kwargs', 'map_to_device',
           'multi_host_barrier', 'override_args_kwargs', 'positional_args_names', 'Renamer',
           're_sign', 'repr_function', 'to_interpolate', 'to_padding', 'to_tuple']

import contextlib
import functools
import inspect
import itertools
import re
from numbers import Number
from typing import Callable, List, Union, Tuple, Iterable, Dict, Pattern, Optional, Sequence

import jax
import jax.numpy as jn
import numpy as np
from jax.interpreters.pxla import ShardedDeviceArray

from objax.constants import ConvPadding, Interpolate
from objax.typing import ConvPaddingInt

CLASS_MODULES = {
    'objax.dpsgd.gradient': 'objax.dpsgd',
    'objax.gradient': 'objax',
    'objax.module': 'objax',
    'objax.nn.layers': 'objax.nn',
    'objax.random.random': 'objax.random',
    'objax.variable': 'objax',
}


class EasyDict(dict):
    """Custom dictionary that allows to access dict values as attributes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class Renamer:
    """Helper class for renaming string contents."""

    def __init__(self,
                 rules: Union[Dict[str, str], Sequence[Tuple[Pattern[str], str]], Callable[[str], str]],
                 chain: Optional['Renamer'] = None):
        """Create a renamer object.

        Args:
            rules: the replacement mapping.
            chain: optionally, another renamer to call after this one completes.
        """
        self.chain = chain
        if callable(rules):
            self.subfn = rules
        elif isinstance(rules, dict):
            regex = re.compile('(%s)' % '|'.join(map(re.escape, rules.keys())))
            self.subfn = functools.partial(regex.sub, lambda m: rules[m.group(0)])
        else:
            def sequence_rename(x):
                for regex, repl in rules:
                    x = regex.sub(repl, x)
                return x

            self.subfn = sequence_rename

    def __call__(self, s: str) -> str:
        """Rename input string `s` using the rules provided to the constructor."""
        news = self.subfn(s)
        return self.chain(news) if self.chain else news


def args_indexes(f: Callable, args: Iterable[str]) -> Iterable[int]:
    """Returns the indexes of variable names of a function."""
    d = {name: i for i, name in enumerate(positional_args_names(f))}
    for name in args:
        index = d.get(name)
        if index is None:
            raise ValueError(f'Function {f} does not have argument of name {name}', (f, name))
        yield index


def class_name(x) -> str:
    """Returns the simplified full name of a class instance."""
    m = x.__class__.__module__
    m = CLASS_MODULES.get(m, m)
    if m.startswith('objax.optimizer'):
        m = 'objax.optimizer'
    return f'{m}.{x.__class__.__name__}'


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


def override_args_kwargs(f: Callable, args: Iterable, kwargs: dict, new_kwargs: dict) -> Tuple[List, dict]:
    """Overrides positional and keyword arguments according to signature of the function using new keyword arguments.

    Args:
        f: callable, which signature is used to determine how to override arguments.
        args: original values of positional arguments.
        kwargs: original values of keyword arguments.
        new_kwargs: new keyword arguments, their values will override original arguments.

    Return:
        args: updated list of positional arguments.
        kwargs: updated dictionary of keyword arguments.
    """
    args = list(args)
    new_kwargs = new_kwargs.copy()
    p = inspect.signature(f).parameters
    for idx, (k, v) in enumerate(itertools.islice(p.items(), len(args))):
        if v.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            break
        if k in new_kwargs:
            args[idx] = new_kwargs.pop(k)
    return args, {**kwargs, **new_kwargs}


def positional_args_names(f: Callable) -> List[str]:
    """Returns the ordered names of the positional arguments of a function."""
    return list(p.name for p in inspect.signature(f).parameters.values()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD))


def to_interpolate(interpolate: Union[Interpolate, str]) -> Union[str]:
    """Expand to a string method for interpolation"""
    if isinstance(interpolate, Interpolate):
        return interpolate.value
    if isinstance(interpolate, str):
        return Interpolate[interpolate.upper()].value

    assert isinstance(interpolate, (str, Interpolate)), f'Argument "{interpolate}" must be a string or Interpolate'


def re_sign(f: Callable) -> Callable:
    """Decorator to replace the signature of an operation with the one from f."""

    def wrap(op):
        op.__signature__ = inspect.signature(f)
        return op

    return wrap


def repr_function(f: Callable) -> str:
    """Human readable function representation."""
    signature = inspect.signature(f)
    args = [f'{k}={v.default}' for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty]
    args = ', '.join(args)
    while not hasattr(f, '__name__'):
        if not hasattr(f, 'func'):
            break
        f = f.func  # Handle functools.partial
    if not hasattr(f, '__name__') and hasattr(f, '__class__'):
        return f.__class__.__name__
    if args:
        return f'{f.__name__}(*, {args})'
    return f.__name__


def to_padding(padding: Union[ConvPadding, str, ConvPaddingInt], ndim: int) \
        -> Union[str, Tuple[Tuple[int, int], ...]]:
    """Expand to a string or a ndim-dimensional tuple of pairs usable for padding."""
    if isinstance(padding, ConvPadding):
        return padding.value
    if isinstance(padding, str):
        return ConvPadding[padding.upper()].value
    if isinstance(padding, int):
        return tuple([(padding, padding)] * ndim)
    if isinstance(padding, tuple) and list(map(type, padding)) == [int, int]:
        return tuple([padding] * ndim)
    return tuple(padding)


def to_tuple(v: Union[Tuple[Number, ...], Number, Iterable], n: int):
    """Converts input to tuple."""
    if isinstance(v, tuple):
        return v
    elif isinstance(v, Number):
        return (v,) * n
    else:
        return tuple(v)
