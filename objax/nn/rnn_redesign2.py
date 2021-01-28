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

from typing import Callable, Tuple, Union

import jax.numpy as jn

import objax
from objax.typing import JaxArray


class MyRnnCell(objax.Module):
    """ Simple RNN cell."""

    def __init__(self, nin: int, nstate: int, activation: Callable = objax.functional.tanh):
        """Creates a MyRnnCell instance.

        Args:
            nin: dimension  of the input tensor.
            state: hidden state tensor has dimensions ``nin`` by ``nstate``.
            activation: activation function for the hidden state layer.
        """
        self.op = objax.nn.Sequential([objax.nn.Linear(nin + nstate, nstate),
                                       objax.nn.Linear(nstate, nstate), activation])

    def __call__(self, x: JaxArray, state: JaxArray) -> JaxArray:
        """Updates and returns hidden state based on input sequence ``x``and input ``state``."""
        return self.op(jn.concatenate((x, state), axis=1))


class DDLRnnCell(objax.Module):
    """ Another simple RNN cell."""

    def __init__(self, nin: int, nstate: int, activation: Callable = objax.functional.tanh):
        """ Creates a DDLRnnCell instance.

        Args:
            nin: dimension of the input tensor.
            nstate: hidden state tensor has dimension ``nin`` by ``nstate``.
            activation: activation function for the hidden state layer.
        """
        self.wxh = objax.nn.Linear(nin, nstate, use_bias=False)
        self.whh = objax.nn.Linear(nstate, nstate)
        self.activation = activation

    def __call__(self, x: JaxArray, state: JaxArray) -> JaxArray:
        """Updates and returns hidden state based on input sequence ``x`` and input ``state``."""
        return self.activation(self.whh(state) + self.wxh(x))


def output_layer(nstate: int, nout: int):
    return objax.nn.Linear(nstate, nout)


class RNN(objax.Module):
    """Simple Recurrent Neural Network (RNN).

    The RNN cell provided as input updates the network's state while the provided output layer generates
    the network's output.
    """

    def __init__(self, cell: objax.Module, output_layer: Union[objax.Module, Callable]):
        """Creates an RNN instance.

        Args:
            cell: RNN cell.
            output_layer: output layer can be a function or another module.
        """
        self.cell = cell
        self.output_layer = output_layer  # Is it better inside or outside?

    def single(self, x_i: JaxArray, state_i: JaxArray) -> Tuple[JaxArray, JaxArray]:
        """Execute one step of the RNN.

        Args:
            x_i: input.
            state_i: current state.

        Returns:
            next output and next state.
        """
        next_state = self.cell(x_i, state_i)
        print("next_state.shape", next_state.shape)
        next_output = self.output_layer(next_state)
        print("next_output.shape", next_output.shape)
        return next_output, next_state

    def __call__(self, x: JaxArray, state: JaxArray) -> Tuple[JaxArray, JaxArray]:
        """Sequentially processes input to generate output.

        Args:
            x: input tensor with dimensions ``batch_size`` by ``sequence_length`` by  ``nin``
            state: Initial RNN state with dimensions ``batch_size`` by ``nstate``.
        Returns:
            Tuple with output with dimensions ``sequence_length`` by ``batch_size`` by ``nout``,
            where ``nout`` is the output dimension of the output layer (or ``nstate`` if there is no output layer)
            and state.
        """
        return objax.functional.scan(self.single, x.transpose((1, 0, 2)), state)  #outputs, final state

class no_batch_RNN(objax.Module):
    """Simple Recurrent Neural Network (RNN).

    The RNN cell provided as input updates the network's state while the provided output layer generates
    the network's output.
    """

    def __init__(self, cell: objax.Module, output_layer: Union[objax.Module, Callable]):
        """Creates an RNN instance.

        Args:
            cell: RNN cell.
            output_layer: output layer can be a function or another module.
        """
        self.cell = cell
        self.output_layer = output_layer  # Is it better inside or outside?

    def single(self, x_i: JaxArray, state_i: JaxArray) -> Tuple[JaxArray, JaxArray]:
        """Execute one step of the RNN.

        Args:
            x_i: input.
            state_i: current state.

        Returns:
            next state and next output.
        """
        next_state = self.cell(x_i, state_i)
        next_output = self.output_layer(next_state)
        return next_output, next_state

    def __call__(self, x: JaxArray, state: JaxArray) -> Tuple[JaxArray, JaxArray]:
        """Sequentially processes input to generate output.

        Args:
            x: input tensor with dimensions ``sequence_length`` by  ``nin``
            state: Initial RNN state with dimensions ``batch_size`` by ``nstate``.

        Returns:
            Tuple with output with dimensions ``sequence_length`` by ``nout``,
            where ``nout`` is the output dimension of the output layer (or ``nstate`` if there is no output layer)
            and state.
        """
        return objax.functional.scan(self.single, x, state)  #outputs, final state


seq, ns, nin, nout, batch = 7, 10, 3, 4, 64
r = RNN(MyRnnCell(nin, ns), output_layer(ns, nout))
x = objax.random.normal((batch, seq, nin))
s = jn.zeros((batch, ns))

print("x.shape", x.shape)
print("s.shape:", s.shape)

#y1 = r(x, s)

r = RNN(DDLRnnCell(nin, ns), lambda x: x)
y2 = r(x, s)
