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
    def __init__(self, nin: int, state: int, activation: Callable = objax.functional.tanh):
        self.op = objax.nn.Sequential([objax.nn.Linear(nin + state, state), objax.functional.relu,
                                       objax.nn.Linear(state, state), activation])

    def __call__(self, state: JaxArray, x: JaxArray) -> JaxArray:
        return self.op(jn.concatenate((x, state), axis=1))


class DDLRnnCell(objax.Module):
    def __init__(self, nin: int, state: int, activation: Callable = objax.functional.tanh):
        self.wxh = objax.nn.Linear(nin, state, use_bias=False)
        self.whh = objax.nn.Linear(state, state)
        self.activation = activation

    def __call__(self, state: JaxArray, x: JaxArray) -> JaxArray:
        return self.activation(self.whh(state) + self.wxh(x))


def output_layer(state: int, nout: int):
    return objax.nn.Linear(state, nout)


class RNN(objax.Module):
    def __init__(self, cell: objax.Module, output_layer: Union[objax.Module, Callable]):
        self.cell = cell
        self.output_layer = output_layer  # Is it better inside or outside?

    def single(self, state_i: JaxArray, x_i: JaxArray) -> Tuple[JaxArray, JaxArray]:
        next_state = self.cell(state_i, x_i)
        next_output = self.output_layer(next_state)
        return next_state, next_output

    def __call__(self, state: JaxArray, x: JaxArray) -> Tuple[JaxArray, JaxArray]:
        # x = (batch, sequence, nin)    state = (batch, state)
        return objax.functional.scan(self.single, state, x.transpose((1, 0, 2)))  # final state, outputs


seq, ns, nin, nout, batch = 7, 10, 3, 4, 64
r = RNN(MyRnnCell(nin, ns), output_layer(ns, nout))
x = objax.random.normal((batch, seq, nin))
s = jn.zeros((batch, ns))
y1 = r(s, x)

r = RNN(DDLRnnCell(nin, ns), lambda x: x)
y2 = r(s, x)
