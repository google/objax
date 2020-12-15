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

__all__ = ['ForceArgs', 'Function', 'Jit', 'Module', 'ModuleList', 'Parallel', 'Vectorize']

from collections import namedtuple
from typing import Optional, List, Union, Callable, Tuple

import jax
import jax.numpy as jn
from jax.interpreters.pxla import ShardedDeviceArray

from objax.typing import JaxArray
from objax.util import class_name, override_args_kwargs, positional_args_names, repr_function
from objax.variable import BaseVar, RandomState, VarCollection


class Module:
    """A module is a container to associate variables and functions."""

    def vars(self, scope: str = '') -> VarCollection:
        """Collect all the variables (and their names) contained in the module and its submodules.
        Important: Variables and modules stored Python structures such as dict or list are not collected. See ModuleList
        if you need such a feature.

        Args:
            scope: string to prefix to the variable names.
        Returns:
            A VarCollection of all the variables.
        """
        vc = VarCollection()
        scope += f'({self.__class__.__name__}).'
        for k, v in self.__dict__.items():
            if isinstance(v, BaseVar):
                vc[scope + k] = v
            elif isinstance(v, Module):
                if k == '__wrapped__':
                    vc.update(v.vars(scope=scope[:-1]))
                else:
                    vc.update(v.vars(scope=scope + k))
        return vc

    def __call__(self, *args, **kwargs):
        """Optional module __call__ method, typically a forward pass computation for standard primitives."""
        raise NotImplementedError


class ForceArgs(Module):
    """Forces override of arguments of given module."""

    ANY = namedtuple('ANY', ())
    """Token used in `ForceArgs.undo` to indicate undo of all values of specific argument."""

    @staticmethod
    def undo(module: Module, **kwargs):
        """Undo ForceArgs on each submodule of the module. Modifications are done in-place.

        Args:
            module: module for which to undo ForceArgs.
            **kwargs: dictionary of argument overrides to undo.
                `name=val` remove override for value `val` of argument `name`.
                `name=ForceArgs.ANY` remove all overrides of argument `name`.
                If `**kwargs` is empty then all overrides will be undone.
        """
        if isinstance(module, ForceArgs):
            if not kwargs:
                module.forced_kwargs = {}
            else:
                module.forced_kwargs = {k: v for k, v in module.forced_kwargs.items()
                                        if (k not in kwargs) or (kwargs[k] not in (v, ForceArgs.ANY))}
            ForceArgs.undo(module.__wrapped__, **kwargs)
        elif isinstance(module, ModuleList):
            for idx, v in enumerate(module):
                if isinstance(v, Module):
                    ForceArgs.undo(v, **kwargs)
                    if isinstance(v, ForceArgs) and not v.forced_kwargs:
                        module[idx] = v.__wrapped__
        else:
            for k, v in module.__dict__.items():
                if isinstance(v, Module):
                    ForceArgs.undo(v, **kwargs)
                    if isinstance(v, ForceArgs) and not v.forced_kwargs:
                        setattr(module, k, v.__wrapped__)

    def __init__(self, module: Module, **kwargs):
        """Initializes ForceArgs by wrapping another module.

        Args:
            module: module which argument will be overridden.
            kwargs: values of keyword arguments which will be forced to use.
        """
        self.__wrapped__ = module
        self.forced_kwargs = kwargs

    def vars(self, scope: str = '') -> VarCollection:
        """Returns the VarCollection of the wrapped module.

        Args:
            scope: string to prefix to the variable names.
        Returns:
            A VarCollection of all the variables of wrapped module.
        """
        return self.__wrapped__.vars(scope=scope)

    def __call__(self, *args, **kwargs):
        """Calls wrapped module using forced args to override wrapped module arguments."""
        args, kwargs = override_args_kwargs(self.__wrapped__, args, kwargs, self.forced_kwargs)
        return self.__wrapped__(*args, **kwargs)

    def __repr__(self):
        args = ', '.join(f'{k}={repr(v)}' for k, v in self.forced_kwargs.items())
        return f'{class_name(self)}(module={repr_function(self.__wrapped__)}, {args})'


class ModuleList(Module, list):
    """This is a replacement for Python's list that provides a vars() method to return all the variables that it
    contains, including the ones contained in the modules and sub-modules in it."""

    def vars(self, scope: str = '') -> VarCollection:
        """Collect all the variables (and their names) contained in the list and its submodules.

        Args:
            scope: string to prefix to the variable names.
        Returns:
            A VarCollection of all the variables.
        """
        vc = VarCollection()
        scope += f'({self.__class__.__name__})'
        for p, v in enumerate(self):
            if isinstance(v, BaseVar):
                vc[f'{scope}[{p}]'] = v
            elif isinstance(v, Module):
                vc.update(v.vars(scope=f'{scope}[{p}]'))
        return vc

    def __getitem__(self, key: Union[int, slice]):
        value = list.__getitem__(self, key)
        if isinstance(key, slice):
            return ModuleList(value)
        return value

    def __repr__(self):
        def f(x):
            if not isinstance(x, Module) and callable(x):
                return repr_function(x)
            x = repr(x).split('\n')
            x = [x[0]] + ['  ' + y for y in x[1:]]
            return '\n'.join(x)

        entries = '\n'.join(f'  [{i}] {f(x)}' for i, x in enumerate(self))
        return f'{class_name(self)}(\n{entries}\n)'


class Function(Module):
    """Turn a function into a Module by keeping the vars it uses."""

    def __init__(self, f: Callable, vc: VarCollection):
        """Function constructor.

        Args:
            f: the function or the module to represent.
            vc: the VarCollection of variables used by the function.
        """
        if hasattr(f, '__name__'):
            self.vc = VarCollection((f'{{{f.__name__}}}{k}', v) for k, v in vc.items())
        else:
            self.vc = VarCollection(vc)
        self.__wrapped__ = f

    def __call__(self, *args, **kwargs):
        """Call the the function."""
        return self.__wrapped__(*args, **kwargs)

    def vars(self, scope: str = '') -> VarCollection:
        """Return the VarCollection of the variables used by the function."""
        if scope:
            return VarCollection((scope + k, v) for k, v in self.vc.items())
        return VarCollection(self.vc)

    @staticmethod
    def with_vars(vc: VarCollection):
        """Method to use as decorator in function definitions."""

        def from_function(f: Callable):
            return Function(f, vc)

        return from_function

    def __repr__(self):
        return f'{class_name(self)}(f={repr_function(self.__wrapped__)})'


class Jit(Module):
    """JIT (Just-In-Time) module takes a function or a module and compiles it for faster execution."""

    def __init__(self,
                 f: Union[Module, Callable],
                 vc: Optional[VarCollection] = None,
                 static_argnums: Optional[Tuple[int, ...]] = None):
        """Jit constructor.

        Args:
            f: the function or the module to compile.
            vc: the VarCollection of variables used by the function or module. This argument is required for functions.
            static_argnums: tuple of indexes of f's input arguments to treat as static (constants)).
                A new graph is compiled for each different combination of values for such inputs.
        """
        self.static_argnums = static_argnums
        if not isinstance(f, Module):
            if vc is None:
                raise ValueError('You must supply the VarCollection used by the function f.')
            f = Function(f, vc)

        def jit(tensor_list: List[JaxArray], kwargs, *args):
            original_values = self.vc.tensors()
            try:
                self.vc.assign(tensor_list)
                return f(*args, **kwargs), self.vc.tensors()
            finally:
                self.vc.assign(original_values)

        self.vc = f.vars() if vc is None else vc
        self._call = jax.jit(jit, static_argnums=tuple(x + 2 for x in sorted(static_argnums or ())))
        self.__wrapped__ = f

    def __call__(self, *args, **kwargs):
        """Call the compiled version of the function or module."""
        output, changes = self._call(self.vc.tensors(), kwargs, *args)
        self.vc.assign(changes)
        return output

    def __repr__(self):
        return f'{class_name(self)}(f={self.__wrapped__}, static_argnums={self.static_argnums or None})'


class Parallel(Module):
    """Parallel module takes a function or a module and compiles it for running on multiple devices in parallel."""

    def __init__(self,
                 f: Union[Module, Callable],
                 vc: Optional[VarCollection] = None,
                 reduce: Callable[[JaxArray], JaxArray] = jn.concatenate,
                 axis_name: str = 'device',
                 static_argnums: Optional[Tuple[int, ...]] = None):
        """Parallel constructor.

        Args:
            f: the function or the module to compile for parallelism.
            vc: the VarCollection of variables used by the function or module. This argument is required for functions.
            reduce: the function used reduce the outputs from many devices to a single device value.
            axis_name: what name to give to the device dimension, used in conjunction with objax.functional.parallel.
            static_argnums: tuple of indexes of f's input arguments to treat as static (constants)).
                A new graph is compiled for each different combination of values for such inputs.
        """
        if not isinstance(f, Module):
            if vc is None:
                raise ValueError('You must supply the VarCollection used by the function f.')
            f = Function(f, vc)

        def pmap(tensor_list: List[ShardedDeviceArray], random_list: List[ShardedDeviceArray], *args):
            original_values = self.vc.tensors()
            try:
                self.vc.assign(tensor_list)
                self.vc.subset(RandomState).assign(random_list)
                return f(*args), self.vc.tensors()
            finally:
                self.vc.assign(original_values)

        static_argnums = sorted(static_argnums or ())
        self.axis_name = axis_name
        self.ndevices = jax.local_device_count()
        self.reduce = reduce
        self.static_argnums = frozenset(static_argnums)
        self.vc = vc or f.vars()
        self._call = jax.pmap(pmap, axis_name=axis_name, static_broadcasted_argnums=[x + 2 for x in static_argnums])
        self.__wrapped__ = f

    def device_reshape(self, x: JaxArray) -> JaxArray:
        """Utility to reshape an input array in order to broadcast to multiple devices."""
        assert hasattr(x, 'ndim'), f'Expected JaxArray, got {type(x)}. If you are trying to pass a scalar to ' \
                                   f'parallel, first convert it to a JaxArray, for example np.float(0.5)'
        if x.ndim == 0:
            return jn.broadcast_to(x, [self.ndevices])
        assert x.shape[0] % self.ndevices == 0, f'Must be able to equally divide batch {x.shape} among ' \
                                                f'{self.ndevices} devices, but does not go equally.'
        return x.reshape((self.ndevices, x.shape[0] // self.ndevices) + x.shape[1:])

    def __call__(self, *args):
        """Call the compiled function or module on multiple devices in parallel.
        Important: Make sure you call this function within the scope of VarCollection.replicate() statement.
        """
        unreplicated = [k for k, v in self.vc.items()
                        if not isinstance(v.value, (ShardedDeviceArray,
                                                    jax.interpreters.partial_eval.JaxprTracer,
                                                    jax.interpreters.partial_eval.DynamicJaxprTracer))]
        assert not unreplicated, \
            f'Some variables were not replicated: {unreplicated}.' \
            'did you forget to call VarCollection.replicate on them?'

        args = [x if i in self.static_argnums
                else jax.tree_map(self.device_reshape, [x])[0] for i, x in enumerate(args)]
        output, changes = self._call(self.vc.tensors(), self.vc.subset(RandomState).tensors(), *args)
        self.vc.assign(changes)
        return jax.tree_map(self.reduce, output)

    def __repr__(self):
        args = dict(f=self.__wrapped__, reduce=repr_function(self.reduce), axis_name=repr(self.axis_name),
                    static_argnums=tuple(sorted(self.static_argnums)) or None)
        args = ', '.join(f'{k}={v}' for k, v in args.items())
        return f'{class_name(self)}({args})'


class Vectorize(Module):
    """Vectorize module takes a function or a module and compiles it for running in parallel on a single device."""

    def __init__(self,
                 f: Union[Module, Callable],
                 vc: Optional[VarCollection] = None,
                 batch_axis: Tuple[Optional[int], ...] = (0,)):
        """Vectorize constructor.

        Args:
            f: the function or the module to compile for vectorization.
            vc: the VarCollection of variables used by the function or module. This argument is required for functions.
            batch_axis: tuple of int or None for each of f's input arguments: the axis to use as batch during
                vectorization. Use None to automatically broadcast.
        """
        if not isinstance(f, Module):
            if vc is None:
                raise ValueError('You must supply the VarCollection used by the function f.')
            f = Function(f, vc)

        def vmap(tensor_list: List[JaxArray], random_list: List[JaxArray], *args):
            original_values = self.vc.tensors()
            try:
                self.vc.assign(tensor_list)
                self.vc.subset(RandomState).assign(random_list)
                return f(*args), self.vc.tensors()
            finally:
                self.vc.assign(original_values)

        fargs = positional_args_names(f)
        assert len(batch_axis) >= len(fargs), f'The batched argument must be specified for all of {f} arguments {fargs}'
        self.batch_axis = batch_axis
        self.batch_axis_argnums = [(x, v) for x, v in enumerate(batch_axis) if v is not None]
        assert self.batch_axis_argnums, f'No arguments to function {f} are vectorizable'
        self.vc = vc or f.vars()
        self._call = jax.vmap(vmap, (None, 0) + batch_axis)
        self.__wrapped__ = f

    def __call__(self, *args):
        """Call the vectorized version of the function or module."""
        assert len(args) == len(self.batch_axis), f'Number of arguments passed {len(args)} must match ' \
                                                  f'batched {len(self.batch_axis)}'
        nsplits = args[self.batch_axis_argnums[0][0]].shape[self.batch_axis_argnums[0][1]]
        output, changes = self._call(self.vc.tensors(), [v.split(nsplits) for v in self.vc.subset(RandomState)], *args)
        for v, u in zip(self.vc, changes):
            v.reduce(u)
        return output

    def __repr__(self):
        return f'{class_name(self)}(f={self.__wrapped__}, batch_axis={self.batch_axis})'
