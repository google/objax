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

__all__ = ['cross_entropy_logits', 
           'cross_entropy_logits_sparse', 
           'l2', 
           'sigmoid_cross_entropy_logits', 
           'mean_squared_error', 
           'mean_absolute_error']

from typing import Union

import jax.numpy as jn

from objax.functional.core import logsumexp
from objax.typing import JaxArray


def cross_entropy_logits(logits: JaxArray, labels: JaxArray) -> JaxArray:
    """Computes the softmax cross-entropy loss on n-dimensional data.

    Args:
        logits: (batch, ..., #class) tensor of logits.
        labels: (batch, ..., #class) tensor of label probabilities (e.g. labels.sum(axis=-1) must be 1)

    Returns:
        (batch, ...) tensor of the cross-entropies for each entry.
    """
    return logsumexp(logits, axis=-1) - (logits * labels).sum(-1)


def cross_entropy_logits_sparse(logits: JaxArray, labels: Union[JaxArray, int]) -> JaxArray:
    """Computes the softmax cross-entropy loss.

    Args:
        logits: (batch, ..., #class) tensor of logits.
        labels: (batch, ...) integer tensor of label indexes in {0, ...,#nclass-1} or just a single integer.

    Returns:
        (batch,) tensor of the cross-entropies for each entry.
    """
    return logsumexp(logits, axis=1) - logits[jn.arange(logits.shape[0]), labels]


def l2(x: JaxArray) -> JaxArray:
    """Computes the L2 loss.

    Args:
        x: n-dimensional tensor of floats.

    Returns:
        scalar tensor containing the l2 loss of x.
    """
    return 0.5 * (x ** 2).sum()


def sigmoid_cross_entropy_logits(logits: JaxArray, labels: Union[JaxArray, int]) -> JaxArray:
    """Computes the sigmoid cross-entropy loss.

    Args:
        logits: (batch, ..., #class) tensor of logits.
        labels: (batch, ..., #class) tensor of label probabilities (e.g. labels.sum(axis=-1) must be 1)

    Returns:
        (batch, ...) tensor of the cross-entropies for each entry.
    """
    return jn.maximum(logits, 0) - logits * labels + jn.log(1 + jn.exp(-jn.abs(logits)))


def mean_squared_error(logits: JaxArray, labels: JaxArray) -> JaxArray:
    """Computes the mean squared error between logits and labels.

    Args:
        logits: (batch_size, d0, .. dN) tensor of the predicted values.
        labels: (batch_size, d0, .. dN) tensor of the ground truth values.

    Returns:
        (batch_size, d0, .. dN-1) tensor of mean squared error for each entry.
    """
    return jn.mean(jn.square(labels - logits), axis=-1)


def mean_absolute_error(logits: JaxArray, labels: JaxArray) -> JaxArray:
    """Computes the mean absolute error between logits and labels.

    Args:
        logits: (batch_size, d0, .. dN) tensor of the predicted values.
        labels: (batch_size, d0, .. dN) tensor of the ground truth values.

    Returns:
        (batch_size, d0, .. dN-1) tensor of the mean absolute error for each entry.
    """
    return jn.mean(jn.abs(labels - logits), axis=-1)
