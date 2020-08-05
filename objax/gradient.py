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

__all__ = ['GradValues']

from typing import List, Optional, Callable, Tuple

import jax

from objax.module import ModuleWrapper
from objax.typing import JaxArray
from objax.variable import BaseState, TrainVar, VarCollection


class GradValues(ModuleWrapper):
    """The GradValues module is used to compute the gradients of a function."""

    def __init__(self, f: Callable,
                 variables: Optional[VarCollection],
                 input_argnums: Optional[Tuple[int, ...]] = None):
        """Constructs an instance to compute the gradient of f w.r.t. variables.

        Args:
            f: the function for which to compute gradients.
            variables: the variables for which to compute gradients.
            input_argnums: input indexes, if any, on which to compute gradients.
        """
        variables = variables or VarCollection()
        super().__init__(variables)
        self.input_argnums = input_argnums or tuple()

        def f_func(inputs_and_train_tensors: List[JaxArray],
                   state_tensors: List[JaxArray],
                   list_args: List):
            inputs = inputs_and_train_tensors[:len(self.input_argnums)]
            train_tensors = inputs_and_train_tensors[len(self.input_argnums):]
            original_vc = self.vc.tensors()
            self.vc.subset(TrainVar).assign(train_tensors)
            self.vc.subset(BaseState).assign(state_tensors)
            for i, arg in zip(self.input_argnums, inputs):
                list_args[i] = arg
            outputs = f(*list_args)
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
            return_value = outputs[0], (outputs, variables.tensors(BaseState))
            self.vc.assign(original_vc)
            return return_value

        self.f = jax.grad(f_func, has_aux=True)

    def __call__(self, *args):
        """Returns the computed gradients for the first value returned by `f` and the values returned by `f`.

        Returns:
            A tuple (gradients , values of f]), where gradients is a list containing
                the input gradients, if any, followed by the variable gradients."""
        inputs = [args[i] for i in self.input_argnums]
        g, (outputs, changes) = self.f(inputs + self.vc.subset(TrainVar).tensors(),
                                       self.vc.subset(BaseState).tensors(),
                                       list(args))
        self.vc.subset(BaseState).assign(changes)
        return g, outputs
