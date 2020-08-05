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

__all__ = ['gain_leaky_relu', 'kaiming_normal', 'kaiming_normal_gain', 'kaiming_truncated_normal', 'truncated_normal',
           'xavier_normal', 'xavier_truncated_normal']

from typing import Tuple

import numpy as np
import scipy.stats

from objax import random
from objax.typing import JaxArray


# Expect format for init APIs
# Convolution: HWIO (number of output channels at the end)
# Linear: IO (number of output dimensions at the end)
# In general: the last dimensions is the number of output channels/dimensions.

def gain_leaky_relu(relu_slope: float = 0.1):
    """The recommended gain value for leaky_relu.

    Args:
        relu_slope: negative slope of leaky_relu.

    Returns:
        The recommended gain value for leaky_relu.
    """
    return np.sqrt(2 / (1 + relu_slope ** 2))


def kaiming_normal(shape: Tuple[int, ...], gain: float = 1) -> JaxArray:
    """Returns a tensor with values assigned using Kaiming He normal initializer from
    `Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
    <https://arxiv.org/abs/1502.01852>`_.

    Args:
        shape: shape of the output tensor.
        gain: optional scaling factor.

    Returns:
        Tensor initialized with normal random variables with standard deviation (gain * kaiming_normal_gain).
    """
    return random.normal(shape, stddev=gain * kaiming_normal_gain(shape))


def kaiming_normal_gain(shape: Tuple[int, ...]) -> float:
    """Returns Kaiming He gain from
    `Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
    <https://arxiv.org/abs/1502.01852>`_.

    Args:
        shape: shape of the output tensor.

    Returns:
        Scalar, the standard deviation gain.
    """
    fan_in = np.prod(shape[:-1])
    return np.sqrt(1 / fan_in)


def kaiming_truncated_normal(shape: Tuple[int, ...], lower: float = -2, upper: float = 2, gain: float = 1) -> JaxArray:
    """Returns a tensor with values assigned using Kaiming He truncated normal initializer from
    `Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
    <https://arxiv.org/abs/1502.01852>`_.

    Args:
        shape: shape of the output tensor.
        lower: lower truncation of the normal.
        upper: upper truncation of the normal.
        gain: optional scaling factor.

    Returns:
        Tensor initialized with truncated normal random variables with standard
        deviation (gain * kaiming_normal_gain) and support [lower, upper].
    """
    truncated_std = scipy.stats.truncnorm.std(a=lower, b=upper, loc=0., scale=1)
    stddev = gain * kaiming_normal_gain(shape) / truncated_std
    return random.truncated_normal(shape, stddev=stddev, lower=lower, upper=upper)


def xavier_normal(shape: Tuple[int, ...], gain: float = 1) -> JaxArray:
    """Returns a tensor with values assigned using Xavier Glorot normal initializer from
    `Understanding the difficulty of training deep feedforward neural networks
    <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_.
    
    Args:
        shape: shape of the output tensor.
        gain: optional scaling factor.

    Returns:
        Tensor initialized with normal random variables with standard deviation (gain * xavier_normal_gain).
    """
    return random.normal(shape, stddev=gain * xavier_normal_gain(shape))


def truncated_normal(shape: Tuple[int, ...], lower: float = -2, upper: float = 2, stddev: float = 1) -> JaxArray:
    """Returns a tensor with values assigned using truncated normal initialization.

    Args:
        shape: shape of the output tensor.
        lower: lower truncation of the normal.
        upper: upper truncation of the normal.
        stddev: expected standard deviation.

    Returns:
        Tensor initialized with truncated normal random variables with standard
        deviation stddev and support [lower, upper].
    """
    truncated_std = scipy.stats.truncnorm.std(a=lower, b=upper, loc=0., scale=1)
    stddev /= truncated_std
    return random.truncated_normal(shape, stddev=stddev, lower=lower, upper=upper)


def xavier_normal_gain(shape: Tuple[int, ...]) -> float:
    """Returns Xavier Glorot gain from
    `Understanding the difficulty of training deep feedforward neural networks
    <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_.

    Args:
        shape: shape of the output tensor.

    Returns:
        Scalar, the standard deviation gain.
    """
    fan_in, fan_out = np.prod(shape[:-1]), shape[-1]
    return np.sqrt(2 / (fan_in + fan_out))


def xavier_truncated_normal(shape: Tuple[int, ...], lower: float = -2, upper: float = 2, gain: float = 1) -> JaxArray:
    """Returns a tensor with values assigned using Xavier Glorot truncated normal initializer from
    `Understanding the difficulty of training deep feedforward neural networks
    <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_.

    Args:
        shape: shape of the output tensor.
        lower: lower truncation of the normal.
        upper: upper truncation of the normal.
        gain: optional scaling factor.

    Returns:
        Tensor initialized with truncated normal random variables with standard
        deviation (gain * xavier_normal_gain) and support [lower, upper].
    """
    truncated_std = scipy.stats.truncnorm.std(a=lower, b=upper, loc=0., scale=1)
    stddev = gain * xavier_normal_gain(shape) / truncated_std
    return random.truncated_normal(shape, stddev=stddev, lower=lower, upper=upper)
