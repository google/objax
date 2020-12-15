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

__all__ = ['BaseVar', 'BaseState', 'RandomState', 'TrainRef', 'StateVar', 'TrainVar', 'VarCollection']

import abc
import re
from contextlib import contextmanager
from typing import List, Union, Tuple, Optional, Iterable, Dict, Iterator, Callable

import jax
import jax.numpy as jn
import jax.random as jr
import numpy as np

from objax.typing import JaxArray
from objax.util import map_to_device, Renamer, repr_function, class_name
from objax.util.check import assert_assigned_type_and_shape_match


def reduce_mean(x: JaxArray) -> JaxArray:
    return x.mean(0)


class BaseVar(abc.ABC):
    """The abstract base class to represent objax variables."""

    def __init__(self, reduce: Optional[Callable[[JaxArray], JaxArray]]):
        """Constructor for BaseVar class.

        Args:
            reduce: a function that takes an array of shape ``(n, *dims)`` and returns one of shape ``(*dims)``. Used to
                    combine the multiple states produced in an objax.Vectorize or an objax.Parallel call.
        """
        self._reduce = reduce

    @property
    @abc.abstractmethod
    def value(self) -> JaxArray:
        raise NotImplementedError('Pure method')

    @value.setter
    @abc.abstractmethod
    def value(self, tensor: JaxArray):
        raise NotImplementedError('Pure method')

    def assign(self, tensor: JaxArray, check=True):
        """Sets the value of the variable."""
        if check:
            assert_assigned_type_and_shape_match(self.value, tensor)
        self.value = tensor

    def reduce(self, tensors: JaxArray):
        """Method called by Parallel and Vectorize to reduce a multiple-device (or batched in case of vectoriaation)
        value to a single device."""
        if self._reduce:
            self.assign(self._reduce(tensors), check=False)

    def __repr__(self):
        rvalue = re.sub('[\n]+', '\n', repr(self._value))
        t = f'{class_name(self)}({rvalue})'
        if not self._reduce:
            return t
        return f'{t[:-1]}, reduce={repr_function(self._reduce)})'


class TrainVar(BaseVar):
    """A trainable variable."""

    def __init__(self, tensor: JaxArray, reduce: Optional[Callable[[JaxArray], JaxArray]] = reduce_mean):
        """TrainVar constructor.

        Args:
            tensor: the initial value of the TrainVar.
            reduce: a function that takes an array of shape ``(n, *dims)`` and returns one of shape ``(*dims)``. Used to
                    combine the multiple states produced in an objax.Vectorize or an objax.Parallel call.
        """
        self._value = tensor
        super().__init__(reduce)

    @property
    def value(self) -> JaxArray:
        """The value is read only as a safety measure to avoid accidentally making TrainVar non-differentiable.
        You can write a value to a TrainVar by using assign."""
        return self._value

    @value.setter
    def value(self, tensor: JaxArray):
        raise ValueError('Direct assignment not allowed, use TrainRef to update a TrainVar.')

    def assign(self, tensor: JaxArray, check=True):
        if check:
            assert_assigned_type_and_shape_match(self.value, tensor)
        self._value = tensor


class BaseState(BaseVar):
    """The abstract base class used to represent objax state variables. State variables are not trainable."""

    def reduce(self, tensors: JaxArray):
        if self._reduce:
            self.assign(self._reduce(tensors), check=False)


class TrainRef(BaseState):
    """A state variable that references a trainable variable for assignment.

     TrainRef are used by optimizers to keep references to trainable variables. This is necessary to differentiate
     them from the optimizer own training variables if any."""

    def __init__(self, ref: TrainVar):
        """TrainRef constructor.

        Args:
            ref: the TrainVar to keep the reference of.
        """
        self.ref = ref
        super().__init__(None)

    @property
    def value(self) -> JaxArray:
        """The value stored in the referenced TrainVar, it can be read or written."""
        return self.ref.value

    @value.setter
    def value(self, tensor: JaxArray):
        self.ref.assign(tensor)

    def __repr__(self):
        return f'{class_name(self)}(ref={repr(self.ref)})'


class StateVar(BaseState):
    """StateVar are variables that get updated manually, and are not automatically updated by optimizers.
    For example, the mean and variance statistics in BatchNorm are StateVar."""

    def __init__(self, tensor: JaxArray, reduce: Optional[Callable[[JaxArray], JaxArray]] = reduce_mean):
        """StateVar constructor.

        Args:
            tensor: the initial value of the StateVar.
            reduce: a function that takes an array of shape ``(n, *dims)`` and returns one of shape ``(*dims)``.
                    Used to combine the multiple states produced in an objax.Vectorize or an objax.Parallel call.
        """
        self._value = tensor
        super().__init__(reduce)

    @property
    def value(self) -> JaxArray:
        """The value stored in the StateVar, it can be read or written."""
        return self._value

    @value.setter
    def value(self, tensor: JaxArray):
        self._value = tensor


class RandomState(StateVar):
    """RandomState are variables that track the random generator state. They are meant to be used internally.
    Currently only the random.Generator module uses them."""

    def __init__(self, seed: int):
        """RandomState constructor.

        Args:
            seed: the initial seed of the random number generator.
        """
        super().__init__(jr.PRNGKey(seed), None)

    def seed(self, seed: int):
        """Sets a new random seed.

        Args:
            seed: the new initial seed of the random number generator.
        """
        self.value = jr.PRNGKey(seed)

    def split(self, n: int) -> List[JaxArray]:
        """Create multiple seeds from the current seed. This is used internally by Parallel and Vectorize to ensure
        that random numbers are different in parallel threads.

        Args:
            n: the number of seeds to generate.
        """
        values = jr.split(self.value, n + 1)
        self._value = values[0]
        return values[1:]


class VarCollection(Dict[str, BaseVar]):
    """A VarCollection is a dictionary (name, var) with some additional methods to make manipulation of collections of
    variables easy. A VarCollection is ordered by insertion order. It is the object returned by Module.vars() and used
    as input by many modules: optimizers, Jit, etc..."""

    def __add__(self, other: 'VarCollection') -> 'VarCollection':
        """Overloaded add operator to merge two VarCollections together."""
        vc = VarCollection(self)
        vc.update(other)
        return vc

    def __iter__(self) -> Iterator[BaseVar]:
        """Create an iterator that iterates over the variables (dict values) and visit them only once.
        If a variable has two names, for example in the case of weight sharing, this iterator yields the variable only
        once."""
        seen = set()
        for v in self.values():
            if v not in seen:
                seen.add(v)
                yield v

    def __setitem__(self, key: str, value: BaseVar):
        """Overload bracket assignment to catch potential conflicts during assignment."""
        if key in self and self[key] != value:
            raise ValueError('Name conflicts when appending to VarCollection', key)
        dict.__setitem__(self, key, value)

    def update(self, other: Union['VarCollection', Iterable[Tuple[str, BaseVar]]]):
        """Overload dict.update method to catch potential conflicts during assignment."""
        if not isinstance(other, self.__class__):
            other = list(other)
        else:
            other = other.items()
        conflicts = set()
        for k, v in other:
            if k in self:
                if self[k] != v:
                    conflicts.add(k)
            else:
                self[k] = v
        if conflicts:
            raise ValueError(f'Name conflicts when combining VarCollection {sorted(conflicts)}')

    def assign(self, tensors: List[JaxArray]):
        """Assign tensors to the variables in the VarCollection. Each variable is assigned only once and in the order
        following the iter(self) iterator.

        Args:
            tensors: the list of tensors used to update variables values.
        """
        vl = list(self)
        assert len(vl) == len(tensors), f'Failed to assign a list with {len(tensors)} tensors to variables with ' \
                                        f'length {len(vl)}.'
        for var, tensor in zip(vl, tensors):
            var.assign(tensor)

    def __str__(self, max_width=100):
        """Pretty print the contents of the VarCollection."""
        text = []
        total = count = 0
        longest_string = max((len(x) for x in self.keys()), default=20)
        longest_string = min(max_width, max(longest_string, 20))
        for name, v in self.items():
            size = np.prod(v.value.shape) if v.value.ndim else 1
            total += size
            count += 1
            text.append(f'{name:{longest_string}} {size:8d} {v.value.shape}')
        text.append(f'{f"+Total({count})":{longest_string}} {total:8d}')
        return '\n'.join(text)

    def rename(self, renamer: Renamer):
        """Rename the entries in the VarCollection."""
        return VarCollection({renamer(k): v for k, v in self.items()})

    @contextmanager
    def replicate(self):
        """A context manager to use in a with statement that replicates the variables in this collection to multiple
        devices. This is used typically prior to call to objax.Parallel, so that all variables have a copy on each
        device.
        Important: replicating also updates the random state in order to have a new one per device.
        """
        ndevices = jax.local_device_count()
        replicated, saved_states = [], []
        for v in self:
            if isinstance(v, RandomState):
                replicated.append(v.split(ndevices))
                saved_states.append(v.value)
            else:
                replicated.append(jn.broadcast_to(v.value, (ndevices,) + v.value.shape))
        replicated = map_to_device(replicated)
        self.assign(replicated)
        yield
        visited = set()
        saved_states.reverse()
        for k, v in self.items():
            if isinstance(v, TrainRef):
                v = v.ref
                assert not isinstance(v, TrainRef)
            if v not in visited:  # Careful not to reduce twice in case of a variable and a reference to it.
                if isinstance(v, RandomState):
                    v.assign(saved_states.pop())
                else:
                    v.reduce(v.value)
                visited.add(v)

    def subset(self, is_a: Optional[Union[type, Tuple[type, ...]]] = None,
               is_not: Optional[Union[type, Tuple[type, ...]]] = None) -> 'VarCollection':
        """Return a new VarCollection that is a filtered subset of the current collection.

        Args:
            is_a: either a variable type or a list of variables types to include.
            is_not: either a variable type or a list of variables types to exclude.
        Returns:
            A new VarCollection containing the subset of variables.
        """
        vc = VarCollection()
        if is_a and is_not:
            vc.update((name, v) for name, v in self.items() if isinstance(v, is_a) and not isinstance(v, is_not))
        elif is_a:
            vc.update((name, v) for name, v in self.items() if isinstance(v, is_a))
        elif is_not:
            vc.update((name, v) for name, v in self.items() if not isinstance(v, is_not))
        return vc

    def tensors(self, is_a: Optional[Union[type, Tuple[type, ...]]] = None) -> List[JaxArray]:
        """Return the list of values for this collection. Similarly to the assign method, each variable value is
        reported only once and in the order following the iter(self) iterator.

        Args:
            is_a: either a variable type or a list of variables types to include.
        Returns:
            A new VarCollection containing the subset of variables.
        """
        if is_a:
            return [x.value for x in self if isinstance(x, is_a)]
        return [x.value for x in self]
