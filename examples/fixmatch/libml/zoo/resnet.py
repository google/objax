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

__all__ = ['ResNetBlock', 'ResNet']

import functools
from typing import Callable

import jax

import objax
from objax.typing import JaxArray


def leaky_relu(x):
    return objax.functional.leaky_relu(x, 0.1)


def conv_args(k, f):
    return dict(w_init=functools.partial(objax.random.normal, stddev=objax.functional.rsqrt(0.5 * k * k * f)))


class ResNetBlock(objax.Module):
    def __init__(self, nin: int, nout: int, stride: int = 1, activate_before_residual: bool = False,
                 bn: Callable = objax.nn.BatchNorm2D):
        self.activate_before_residual = activate_before_residual
        self.bn = bn(nin, momentum=0.999)
        self.residual = objax.nn.Sequential([objax.nn.Conv2D(nin, nout, 3, strides=stride, **conv_args(3, nout)),
                                             bn(nout, momentum=0.999), leaky_relu,
                                             objax.nn.Conv2D(nout, nout, 3, **conv_args(3, nout))])
        self.passthrough = objax.nn.Conv2D(nin, nout, 1, strides=stride, **conv_args(1, nout)) if nin != nout else None

    def __call__(self, x: JaxArray, training: bool) -> JaxArray:
        y = leaky_relu(self.bn(x, training))
        if self.activate_before_residual:
            x = y
        if self.passthrough:
            x = self.passthrough(x)
        return x + self.residual(y, training=training)


class ResNet(objax.nn.Sequential):
    @staticmethod
    def mean_reduce(x: JaxArray) -> JaxArray:
        return x.mean((2, 3))

    def __init__(self, nin: int, nclass: int, scales: int, filters: int, repeat: int, dropout: int = 0,
                 bn: Callable = objax.nn.BatchNorm2D, **kwargs):
        del kwargs
        n = 16
        ops = [objax.nn.Conv2D(nin, n, 3, **conv_args(3, n))]
        for scale in range(scales):
            last_n, n = n, filters << scale
            ops.append(ResNetBlock(last_n, n, stride=2 if scale else 1, activate_before_residual=scale == 0, bn=bn))
            ops.extend([ResNetBlock(n, n, bn=bn) for _ in range(repeat - 1)])
        ops.extend([bn(n, momentum=0.999), leaky_relu, self.mean_reduce,
                    objax.nn.Dropout(1 - dropout),
                    objax.nn.Linear(n, nclass, w_init=objax.nn.init.xavier_truncated_normal)])
        super().__init__(ops)
