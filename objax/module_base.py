# Copyright 2021 Google LLC
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

"""Module with implementation of base class for all Objax modules.

It put into separate file to avoid circular dependencies in the code."""

__all__ = ['Module']

from objax.variable import BaseVar, VarCollection


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
