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
           'mean_absolute_error',
           'mean_squared_error',
           'mean_squared_log_error',
           'sigmoid_cross_entropy_logits']

from typing import Union, Iterable, Optional

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
        (batch, ...) tensor of the cross-entropies for each entry.
    """
    if isinstance(labels, int):
        labeled_logits = logits[..., labels]
    else:
        labeled_logits = jn.take_along_axis(logits, labels[..., None], -1).squeeze(-1)

    return logsumexp(logits, axis=-1) - labeled_logits


def l2(x: JaxArray) -> JaxArray:
    """Computes the L2 loss.

    Args:
        x: n-dimensional tensor of floats.

    Returns:
        scalar tensor containing the l2 loss of x.
    """
    return 0.5 * (x ** 2).sum()


def mean_absolute_error(x: JaxArray, y: JaxArray, keep_axis: Optional[Iterable[int]] = (0,)) -> JaxArray:
    """Computes the mean absolute error between x and y.

    Args:
        x: a tensor of shape (d0, .. dN-1).
        y: a tensor of shape (d0, .. dN-1).
        keep_axis: a sequence of the dimensions to keep, use `None` to return a scalar value.

    Returns:
        tensor of shape (d_i, ..., for i in keep_axis) containing the mean absolute error.
    """
    loss = jn.abs(x - y)
    axis = [i for i in range(loss.ndim) if i not in (keep_axis or ())]
    return loss.mean(axis)


def mean_squared_error(x: JaxArray, y: JaxArray, keep_axis: Optional[Iterable[int]] = (0,)) -> JaxArray:
    """Computes the mean squared error between x and y.

    Args:
        x: a tensor of shape (d0, .. dN-1).
        y: a tensor of shape (d0, .. dN-1).
        keep_axis: a sequence of the dimensions to keep, use `None` to return a scalar value.

    Returns:
        tensor of shape (d_i, ..., for i in keep_axis) containing the mean squared error.
    """
    loss = (x - y) ** 2
    axis = [i for i in range(loss.ndim) if i not in (keep_axis or ())]
    return loss.mean(axis)


def mean_squared_log_error(y_true: JaxArray, y_pred: JaxArray, keep_axis: Optional[Iterable[int]] = (0,)) -> JaxArray:
    """Computes the mean squared logarithmic error between y_true and y_pred.

    Args:
        y_true: a tensor of shape (d0, .. dN-1).
        y_pred: a tensor of shape (d0, .. dN-1).
        keep_axis: a sequence of the dimensions to keep, use `None` to return a scalar value.

    Returns:
        tensor of shape (d_i, ..., for i in keep_axis) containing the mean squared error.
    """
    loss = (jn.log1p(y_true) - jn.log1p(y_pred)) ** 2
    axis = [i for i in range(loss.ndim) if i not in (keep_axis or ())]
    return loss.mean(axis)


def sigmoid_cross_entropy_logits(logits: JaxArray, labels: Union[JaxArray, int]) -> JaxArray:
    """Computes the sigmoid cross-entropy loss.

    Args:
        logits: (batch, ..., #class) tensor of logits.
        labels: (batch, ..., #class) tensor of label probabilities (e.g. labels.sum(axis=-1) must be 1)

    Returns:
        (batch, ...) tensor of the cross-entropies for each entry.
    """
    return jn.maximum(logits, 0) - logits * labels + jn.log(1 + jn.exp(-jn.abs(logits)))
