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

"""Module with WideResNet implementation.

See https://arxiv.org/abs/1605.07146 for detail.
"""

__all__ = ['WRNBlock', 'WideResNetGeneral', 'WideResNet']

import functools
from typing import Callable, List

import objax
from objax.typing import JaxArray

BN_MOM = 0.9
BN_EPS = 1e-5


def conv_args(kernel_size: int, nout: int):
    """Returns list of arguments which are common to all convolutions.

    Args:
        kernel_size: size of convolution kernel (single number).
        nout: number of output filters.

    Returns:
        Dictionary with common convoltion arguments.
    """
    stddev = objax.functional.rsqrt(0.5 * kernel_size * kernel_size * nout)
    return dict(w_init=functools.partial(objax.random.normal, stddev=stddev),
                use_bias=False,
                padding=objax.constants.ConvPadding.SAME)


class WRNBlock(objax.Module):
    """WideResNet block."""

    def __init__(self,
                 nin: int,
                 nout: int,
                 stride: int = 1,
                 bn: Callable = functools.partial(objax.nn.BatchNorm2D, momentum=BN_MOM, eps=BN_EPS)):
        """Creates WRNBlock instance.

        Args:
            nin: number of input filters.
            nout: number of output filters.
            stride: stride for convolution and projection convolution in this block.
            bn: module which used as batch norm function.
        """
        if nin != nout or stride > 1:
            self.proj_conv = objax.nn.Conv2D(nin, nout, 1, strides=stride, **conv_args(1, nout))
        else:
            self.proj_conv = None

        self.norm_1 = bn(nin)
        self.conv_1 = objax.nn.Conv2D(nin, nout, 3, strides=stride, **conv_args(3, nout))
        self.norm_2 = bn(nout)
        self.conv_2 = objax.nn.Conv2D(nout, nout, 3, strides=1, **conv_args(3, nout))

    def __call__(self, x: JaxArray, training: bool) -> JaxArray:
        o1 = objax.functional.relu(self.norm_1(x, training))
        y = self.conv_1(o1)
        o2 = objax.functional.relu(self.norm_2(y, training))
        z = self.conv_2(o2)
        return z + self.proj_conv(o1) if self.proj_conv else z + x


class WideResNetGeneral(objax.nn.Sequential):
    """Base WideResNet implementation."""

    @staticmethod
    def mean_reduce(x: JaxArray) -> JaxArray:
        return x.mean((2, 3))

    def __init__(self,
                 nin: int,
                 nclass: int,
                 blocks_per_group: List[int],
                 width: int,
                 bn: Callable = functools.partial(objax.nn.BatchNorm2D, momentum=BN_MOM, eps=BN_EPS)):
        """Creates WideResNetGeneral instance.

        Args:
            nin: number of channels in the input image.
            nclass: number of output classes.
            blocks_per_group: number of blocks in each block group.
            width: multiplier to the number of convolution filters.
            bn: module which used as batch norm function.
        """
        widths = [int(v * width) for v in [16 * (2 ** i) for i in range(len(blocks_per_group))]]

        n = 16
        ops = [objax.nn.Conv2D(nin, n, 3, **conv_args(3, n))]
        for i, (block, width) in enumerate(zip(blocks_per_group, widths)):
            stride = 2 if i > 0 else 1
            ops.append(WRNBlock(n, width, stride, bn))
            for b in range(1, block):
                ops.append(WRNBlock(width, width, 1, bn))
            n = width
        ops += [bn(n),
                objax.functional.relu,
                self.mean_reduce,
                objax.nn.Linear(n, nclass, w_init=objax.nn.init.xavier_truncated_normal)
                ]
        super().__init__(ops)


class WideResNet(WideResNetGeneral):
    """WideResNet implementation with 3 groups.

    Reference:
        http://arxiv.org/abs/1605.07146
        https://github.com/szagoruyko/wide-residual-networks
    """

    def __init__(self,
                 nin: int,
                 nclass: int,
                 depth: int = 28,
                 width: int = 2,
                 bn: Callable = functools.partial(objax.nn.BatchNorm2D, momentum=BN_MOM, eps=BN_EPS)):
        """Creates WideResNet instance.

        Args:
            nin: number of channels in the input image.
            nclass: number of output classes.
            depth: number of convolution layers. (depth-4) should be divisible by 6
            width: multiplier to the number of convolution filters.
            bn: module which used as batch norm function.
        """
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        blocks_per_group = [n] * 3
        super().__init__(nin, nclass, blocks_per_group, width, bn)
