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

from typing import Callable

import jax.numpy as jn

from objax import Module
from objax.nn import Linear
from objax.nn.init import kaiming_normal
from objax.typing import JaxArray
from objax.variable import TrainVar, StateVar


class RNN(Module):
    """ Recurrent Neural Network (RNN) block."""

    def __init__(self,
                 nstate: int,
                 nin: int,
                 nout: int,
                 activation: Callable = jn.tanh,
                 w_init: Callable = kaiming_normal):
        """Creates an RNN instance.

        Args:
            nstate: number of hidden units.
            nin: number of input units.
            nout: number of output units.
            activation: actication function for hidden layer.
            w_init: weight initializer for RNN model weights.
        """
        self.num_inputs = nin
        self.num_outputs = nout
        self.nstate = nstate
        self.activation = activation

        # Hidden layer parameters
        self.w_xh = TrainVar(w_init((self.num_inputs, self.nstate)))
        self.w_hh = TrainVar(w_init((self.nstate, self.nstate)))
        self.b_h = TrainVar(jn.zeros(self.nstate))

        self.output_layer = Linear(self.nstate, self.num_outputs)

    def init_state(self, batch_size):
        """Initialize hidden state for input batch of size ``batch_size``."""
        self.state = StateVar(jn.zeros((batch_size, self.nstate)))

    def __call__(self, inputs: JaxArray, only_return_final=False) -> JaxArray:
        """Forward pass through RNN.

        Args:
            inputs: ``JaxArray`` with dimensions ``num_steps, batch_size, vocabulary_size``.
            only_return_final: return only the last output if ``True``, or all output otherwise.`

        Returns:
            Output tensor with dimensions ``num_steps * batch_size, vocabulary_size``.
        """
        # Dimensions: num_steps, batch_size, vocab_size
        outputs = []
        for x in inputs:
            self.state.value = self.activation(
                jn.dot(x, self.w_xh.value) +
                jn.dot(self.state.value, self.w_hh.value) +
                self.b_h.value)
            y = self.output_layer(self.state.value)
            outputs.append(y)
        if only_return_final: return outputs[-1]
        return jn.concatenate(outputs, axis=0)
