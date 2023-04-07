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

__all__ = ['Grad', 'GradValues', 'Hessian', 'Jacobian']

import inspect
from typing import List, Optional, Callable, Tuple, Dict, Union

import jax

from objax.module import Function, Module
from objax.typing import JaxArray
from objax.util import repr_function, class_name
from objax.variable import BaseState, TrainVar, VarCollection


class _DerivativeBase(Module):
    """Base class for various modules which compute derivatives."""

    def __init__(self,
                 derivative_fn: Callable,
                 f: Union[Module, Callable],
                 variables: Optional[VarCollection],
                 input_argnums: Optional[Tuple[int, ...]] = None,
                 return_all_f_outputs: bool = False):
        """Constructs an instance to compute the derivatives of f w.r.t. variables.
        Args:
            derivative_fn: JAX transformation which computes derivative.
            f: the function for which to compute derivatives.
            variables: the variables for which to compute derivatives.
            input_argnums: input indexes, if any, on which to compute derivatives.
            return_all_f_outputs: if True also return original outputs of the fuction along with derivatives.
        """
        self.f = f
        self.vc = variables = VarCollection(variables or ())
        if not isinstance(f, Module):
            f = Function(f, self.vc)

        assert isinstance(input_argnums, tuple) or input_argnums is None, \
            f'Must pass a tuple of indices to input_argnums; received {input_argnums}.'
        self.input_argnums = input_argnums or tuple()
        self.return_all_f_outputs = return_all_f_outputs

        def f_func(inputs_and_train_tensors: List[JaxArray],
                   state_tensors: List[JaxArray],
                   list_args: List,
                   kwargs: Dict):
            inputs = inputs_and_train_tensors[:len(self.input_argnums)]
            train_tensors = inputs_and_train_tensors[len(self.input_argnums):]
            original_vc = self.vc.tensors()
            try:
                self.vc.subset(TrainVar).assign(train_tensors)
                self.vc.subset(BaseState).assign(state_tensors)
                for i, arg in zip(self.input_argnums, inputs):
                    list_args[i] = arg
                outputs = f(*list_args, **kwargs)
                if not isinstance(outputs, (list, tuple)):
                    outputs = [outputs]
                if self.return_all_f_outputs:
                    return outputs[0], (outputs, variables.tensors())
                else:
                    return outputs[0], variables.tensors()
            finally:
                self.vc.assign(original_vc)

        self._call = derivative_fn(f_func)

    def __call__(self, *args, **kwargs):
        """Returns the computed gradients for the first value returned by `f` and optionally values returned by `f`."""
        inputs = [args[i] for i in self.input_argnums]
        g, aux_out = self._call(inputs + self.vc.subset(TrainVar).tensors(),
                                self.vc.subset(BaseState).tensors(),
                                list(args), kwargs)
        if self.return_all_f_outputs:
            outputs, changes = aux_out
            self.vc.assign(changes)
            return g, outputs
        else:
            changes = aux_out
            self.vc.assign(changes)
            return g

    def vars(self, scope: str = '') -> VarCollection:
        """Return the VarCollection of the variables used."""
        if scope:
            return VarCollection((scope + k, v) for k, v in self.vc.items())
        return VarCollection(self.vc)

    def __repr__(self):
        f = repr(self.f) if isinstance(self.f, Module) else repr_function(self.f)
        return f'{class_name(self)}(f={f}, input_argnums={self.input_argnums or None})'


class GradValues(_DerivativeBase):
    """The GradValues module is used to compute the gradients of a function."""

    def __init__(self,
                 f: Union[Module, Callable],
                 variables: Optional[VarCollection],
                 input_argnums: Optional[Tuple[int, ...]] = None):
        """Constructs an instance to compute the gradient of f w.r.t. variables.

        Args:
            f: the function for which to compute gradients.
            variables: the variables for which to compute gradients.
            input_argnums: input indexes, if any, on which to compute gradients.
        """
        super().__init__(lambda f_func: jax.grad(f_func, has_aux=True),
                         f=f,
                         variables=variables,
                         input_argnums=input_argnums,
                         return_all_f_outputs=True)

        signature = inspect.signature(f)
        self.__wrapped__ = f
        self.__signature__ = signature.replace(return_annotation=Tuple[List[JaxArray],
                                                                       signature.return_annotation])

    def __call__(self, *args, **kwargs):
        """Returns the computed gradients for the first value returned by `f` and the values returned by `f`.
        Returns:
            A tuple (gradients, values of f), where gradients is a list containing
            the input gradients, if any, followed by the variable gradients."""
        return super().__call__(*args, **kwargs)


class Grad(_DerivativeBase):
    """The Grad module is used to compute the gradients of a function."""

    def __init__(self, f: Callable,
                 variables: Optional[VarCollection],
                 input_argnums: Optional[Tuple[int, ...]] = None):
        """Constructs an instance to compute the gradient of f w.r.t. variables.

        Args:
            f: the function for which to compute gradients.
            variables: the variables for which to compute gradients.
            input_argnums: input indexes, if any, on which to compute gradients.
        """
        super().__init__(lambda f_func: jax.grad(f_func, has_aux=True),
                         f=f,
                         variables=variables,
                         input_argnums=input_argnums)
        signature = inspect.signature(f)
        self.__wrapped__ = f
        self.__signature__ = signature.replace(return_annotation=List[JaxArray])

    def __call__(self, *args, **kwargs):
        """Returns the computed gradients for the first value returned by `f`.

        Returns:
            A list of input gradients, if any, followed by the variable gradients."""
        return super().__call__(*args, **kwargs)


class Jacobian(_DerivativeBase):
    """The Jacobian module computes Jacobian matrix of a function."""

    def __init__(self,
                 f: Union[Module, Callable],
                 variables: Optional[VarCollection],
                 input_argnums: Optional[Tuple[int, ...]] = None):
        """Constructs an instance to compute the Jacobian of f w.r.t. variables and arguments.

        Args:
            f: the function for which to compute Jacobian.
            variables: the variables for which to compute gradients.
            input_argnums: input indexes, if any, on which to compute gradients.
        """
        super().__init__(lambda f_func: jax.jacfwd(f_func, has_aux=True),
                         f=f,
                         variables=variables,
                         input_argnums=input_argnums)


class Hessian(_DerivativeBase):
    """The Hessian module computes Hessian of a function."""

    def __init__(self,
                 f: Union[Module, Callable],
                 variables: Optional[VarCollection],
                 input_argnums: Optional[Tuple[int, ...]] = None):
        """Constructs an instance to compute the Hessian of f w.r.t. variables and arguments.

        Args:
            f: the function for which to compute Hessian.
            variables: the variables for which to compute gradients.
            input_argnums: input indexes, if any, on which to compute gradients.
        """
        super().__init__(lambda f_func: jax.jacfwd(jax.jacrev(f_func, has_aux=True), has_aux=True),
                         f=f,
                         variables=variables,
                         input_argnums=input_argnums)
