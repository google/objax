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
from objax.nn.init import kaiming_normal
from objax.typing import JaxArray
from objax.variable import TrainVar, StateVar
from objax.functional import sigmoid


class GRU(Module):
    """ Gated Recurrent Unit (GRU) block."""

    def __init__(self,
                 nstate: int,
                 nin: int,
                 nout: int,
                 w_init: Callable = kaiming_normal):
        """Creates a GRU instance.

        Args:
            nstate: number of hidden units.
            nin: number of input units.
            nout: number of output units.
            w_init: weight initializer for GRU model weights.
        """
        self.num_inputs = nin
        self.num_outputs = nout
        self.nstate = nstate

        # Update gate parameters
        self.w_xz = TrainVar(w_init((self.num_inputs, self.nstate)))
        self.w_hz = TrainVar(w_init((self.nstate, self.nstate)))
        self.b_z = TrainVar(jn.zeros(self.nstate))

        # Reset gate parameters
        self.w_xr = TrainVar(w_init((self.num_inputs, self.nstate)))
        self.w_hr = TrainVar(w_init((self.nstate, self.nstate)))
        self.b_r = TrainVar(jn.zeros(self.nstate))

        # Candidate hidden state parameters
        self.w_xh = TrainVar(w_init((self.num_inputs, self.nstate)))
        self.w_hh = TrainVar(w_init((self.nstate, self.nstate)))
        self.b_h = TrainVar(jn.zeros(self.nstate))

        # Output layer parameters
        self.w_hq = TrainVar(w_init((self.nstate, self.num_outputs)))
        self.b_q = TrainVar(jn.zeros(self.num_outputs))

    def init_state(self, batch_size):
        """Initialize hidden state for input batch of size ``batch_size``."""
        self.state = StateVar(jn.zeros((batch_size, self.nstate)))

    def __call__(self, inputs: JaxArray, only_return_final=False) -> JaxArray:
        """Forward pass through GRU.

        Args:
            inputs: ``JaxArray`` with dimensions ``num_steps, batch_size, vocabulary_size``.
            only_return_final: return only the last output if ``True``, or all output otherwise.`

        Returns:
            Output tensor with dimensions ``num_steps * batch_size, vocabulary_size``.
        """
        # Dimensions: num_steps, batch_size, vocab_size
        outputs = []
        for x in inputs:
            update_gate = sigmoid(jn.dot(x, self.w_xz.value) + jn.dot(self.state.value, self.w_hz.value) +
                                  self.b_z. bnhvalue)
            reset_gate = sigmoid(jn.dot(x, self.w_xr.value) + jn.dot(self.state.value, self.w_hr.value) +
                                 self.b_r.value)
            candidate_state = jn.tanh(jn.dot(x, self.w_xh.value) +
                                      jn.dot(reset_gate * self.state.value, self.w_hh.value) + self.b_h.value)
            self.state.value = update_gate * self.state.value + (1 - update_gate) * candidate_state
            y = jn.dot(self.state.value, self.w_hq.value) + self.b_q.value
            outputs.append(y)
        if only_return_final:
            return outputs[-1]
        return jn.concatenate(outputs, axis=0)
