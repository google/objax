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

<<<<<<< HEAD
__all__ = [
    "cross_entropy_logits",
    "cross_entropy_logits_sparse",
    "mean_squared_logarithmic_error",
    "l2",
    "sigmoid_cross_entropy_logits",
]
=======
__all__ = ['cross_entropy_logits', 
           'cross_entropy_logits_sparse', 
           'l2', 
           'mean_absolute_error',
           'mean_squared_error', 
           'sigmoid_cross_entropy_logits']
>>>>>>> 97ad6998d942f041b7d8cdb9749044cdbc3abcad

from typing import Union, Iterable

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


def cross_entropy_logits_sparse(
    logits: JaxArray, labels: Union[JaxArray, int]
) -> JaxArray:
    """Computes the softmax cross-entropy loss.

    Args:
        logits: (batch, ..., #class) tensor of logits.
        labels: (batch, ...) integer tensor of label indexes in {0, ...,#nclass-1} or just a single integer.

    Returns:
        (batch,) tensor of the cross-entropies for each entry.
    """
    return logsumexp(logits, axis=1) - logits[jn.arange(logits.shape[0]), labels]


def mean_squared_logarithmic_error(y_true: JaxArray, y_pred: JaxArray) -> JaxArray:
    """Computes the the mean squared logarithmic error.

    Args:
        y_true: float tensor of shape (batch, d0, d1...dn)
        y_pred: float tensor of shape (batch, d0, d1...dn)

    Returns:
        float tensor of shape (batch, d0, d1...dn-1)
    """
    y_true_log = jn.log1p(y_true)
    y_pred_log = jn.log1p(y_pred)

    loss = (y_pred_log - y_true_log) ** 2
    return loss.mean(axis=-1)


def l2(x: JaxArray) -> JaxArray:
    """Computes the L2 loss.

    Args:
        x: n-dimensional tensor of floats.

    Returns:
        scalar tensor containing the l2 loss of x.
    """
    return 0.5 * (x ** 2).sum()


<<<<<<< HEAD
def sigmoid_cross_entropy_logits(
    logits: JaxArray, labels: Union[JaxArray, int]
) -> JaxArray:
=======
def mean_absolute_error(x: JaxArray, y: JaxArray, keep_dims: Iterable[int] = (0,)) -> JaxArray:
    """Computes the mean absolute error between x and y.
    
    Args:
        x: a tensor of shape (d0, .. dN-1).
        y: a tensor of shape (d0, .. dN-1).
        keep_dims: a sequence of the dimensions to keep.
        
    Returns:
        (d_i, ..., for i in keep_dims) tensor of the mean absolute error.
    """
    loss = jn.abs(x - y)
    axis = [i for i in range(loss.ndim) if i not in keep_dims]
    return loss.mean(axis)


def mean_squared_error(x: JaxArray, y: JaxArray, keep_dims: Iterable[int] = (0,)) -> JaxArray:
    """Computes the mean squared error between x and y.
    
    Args:
        x: a tensor of shape (d0, .. dN-1).
        y: a tensor of shape (d0, .. dN-1).
        keep_dims: a sequence of the dimensions to keep.
        
    Returns:
        (d_i, ..., for i in keep_dims) tensor of the mean squared error.
    """
    loss = (x - y) ** 2
    axis = [i for i in range(loss.ndim) if i not in keep_dims]
    return loss.mean(axis)


def sigmoid_cross_entropy_logits(logits: JaxArray, labels: Union[JaxArray, int]) -> JaxArray:
>>>>>>> 97ad6998d942f041b7d8cdb9749044cdbc3abcad
    """Computes the sigmoid cross-entropy loss.

    Args:
        logits: (batch, ..., #class) tensor of logits.
        labels: (batch, ..., #class) tensor of label probabilities (e.g. labels.sum(axis=-1) must be 1)

    Returns:
        (batch, ...) tensor of the cross-entropies for each entry.
    """
    return jn.maximum(logits, 0) - logits * labels + jn.log(1 + jn.exp(-jn.abs(logits)))
