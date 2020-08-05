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

__all__ = ['Jit', 'Module', 'ModuleList', 'ModuleWrapper', 'Parallel', 'Vectorize']

from types import MethodType
from typing import Optional, List, Union, Callable, Tuple

import jax
import jax.numpy as jn
from jax.interpreters.pxla import ShardedDeviceArray

from objax.typing import JaxArray
from objax.util import positional_args_names
from objax.variable import BaseState, BaseVar, RandomState, VarCollection


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
                vc.update(v.vars(scope=scope + k))
        return vc

    def __call__(self, *args, **kwargs):
        """Optional module __call__ method, typically a forward pass computation for standard primitives."""
        raise NotImplementedError


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


class ModuleWrapper(Module):
    """Module whose sole purpose is to store a collectable VarCollection. This classs is exclusively
    used internally by Objax, for example in Jit, Vectorize and Parallel."""

    def __init__(self, vc: VarCollection):
        super().__init__()
        self.vc = VarCollection((f'({self.__class__.__name__}){k}', v) for k, v in vc.items())

    def vars(self, scope: str = '') -> VarCollection:
        """Collect all the variables (and their names) contained in the VarCollection.

        Args:
            scope: string to prefix to the variable names.
        Returns:
            A VarCollection of all the variables.
        """
        return VarCollection((scope + k, v) for k, v in self.vc.items())


class Jit(ModuleWrapper):
    """JIT (Just-In-Time) module takes a function or a module and compiles it for faster execution."""

    def __init__(self,
                 f: Union[Module, Callable],
                 vc: Optional[VarCollection] = None,
                 static_argnums: Optional[Tuple[int, ...]] = None):
        """Jit constructor.

        Args:
            f: the function or the module to compile.
            vc: the VarCollection of variables used by the function or module. This argument is equired for functions.
            static_argnums: tuple of indexes of f's input arguments to treat as static (constants)).
                A new graph is compiled for each different combination of values for such inputs.
        """
        if vc is None:
            if not isinstance(f, Module):
                raise ValueError('You must supply the VarCollection used by the function f.')
            vc = f.vars()

        super().__init__(vc)
        self._call = self.jit_local_method(f, sorted(static_argnums or ()))

    def jit_local_method(self, f, static_argnums):
        """Compiles a function or module and returns method that can be attached to self instance.

        Args:
            f: function or module to compile.
            static_argnums: indexes of the arguments to be treated as static.

        Returns:
            A method containing the compiled version of f.
        """

        def jit(tensor_list: List[JaxArray], *args):
            original_values = self.vc.tensors()
            self.vc.assign(tensor_list)
            output = f(*args), self.vc.tensors(BaseState)
            self.vc.assign(original_values)
            return output

        jitf = jax.jit(jit, static_argnums=tuple(x + 1 for x in static_argnums))

        def local_method(self, *args):
            output, changes = jitf(self.vc.tensors(), *args)
            self.vc.subset(BaseState).assign(changes)
            return output

        return MethodType(local_method, self)

    def __call__(self, *args):
        """Call the compiled version of the function or module."""
        return self._call(*args)


class Parallel(ModuleWrapper):
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
        if vc is None:
            if not isinstance(f, Module):
                raise ValueError('You must supply the VarCollection used by the function f.')
            vc = f.vars()

        super().__init__(vc)
        static_argnums = sorted(static_argnums or ())
        self.reduce = reduce
        self.ndevices = jax.device_count()
        self.static_argnums = frozenset(static_argnums)

        def pmap(tensor_list: List[ShardedDeviceArray], random_list: List[ShardedDeviceArray], *args):
            original_values = self.vc.tensors()
            self.vc.assign(tensor_list)
            self.vc.subset(RandomState).assign(random_list)
            output = f(*args), self.vc.tensors(BaseState)
            self.vc.assign(original_values)
            return output

        self._call = jax.pmap(pmap, axis_name=axis_name, static_broadcasted_argnums=[x + 2 for x in static_argnums])

    def device_reshape(self, x: JaxArray) -> JaxArray:
        """Utility to reshape an input array in order to broadcast to multiple devices."""
        return x.reshape((self.ndevices, x.shape[0] // self.ndevices) + x.shape[1:])

    def __call__(self, *args):
        """Call the compiled function or module on multiple devices in parallel.
        Important: Make sure you call this function within the scope of VarCollection.replicate() statement.
        """
        args = [x if i in self.static_argnums else self.device_reshape(x) for i, x in enumerate(args)]
        output, changes = self._call(self.vc.tensors(), self.vc.subset(RandomState).tensors(), *args)
        self.vc.subset(BaseState).assign(changes)
        return jax.tree_map(self.reduce, output)


class Vectorize(ModuleWrapper):
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
        if vc is None:
            if not isinstance(f, Module):
                raise ValueError('You must supply the VarCollection used by the function f.')
            vc = f.vars()

        super().__init__(vc)
        fargs = positional_args_names(f)
        assert len(batch_axis) >= len(fargs), f'The batched argument must be specified for all of {f} arguments {fargs}'
        self.batch_axis = batch_axis
        self.batch_axis_argnums = [(x, v) for x, v in enumerate(batch_axis) if v is not None]
        assert self.batch_axis_argnums, f'No arguments to function {f} are vectorizable'

        def vmap(tensor_list: List[JaxArray], random_list: List[JaxArray], *args):
            original_values = self.vc.tensors()
            self.vc.assign(tensor_list)
            self.vc.subset(RandomState).assign(random_list)
            output = f(*args), self.vc.tensors(BaseState)
            self.vc.assign(original_values)
            return output

        self._call = jax.vmap(vmap, (None, 0) + batch_axis)

    def __call__(self, *args):
        """Call the vectorized version of the function or module."""
        assert len(args) == len(self.batch_axis), f'Number of arguments passed {len(args)} must match ' \
                                                  f'batched {len(self.batch_axis)}'
        nsplits = args[self.batch_axis_argnums[0][0]].shape[self.batch_axis_argnums[0][1]]
        output, changes = self._call(self.vc.tensors(), [v.split(nsplits) for v in self.vc.subset(RandomState)], *args)
        for v, u in zip(self.vc.subset(BaseState), changes):
            v.reduce(u)
        return output
