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

__all__ = ['PrivateGradValues']

import functools
from typing import Optional, Callable, Tuple

import jax
import jax.numpy as jn

from objax import random
from objax.gradient import GradValues
from objax.module import Function, Module, Vectorize
from objax.typing import JaxArray
from objax.util import repr_function, class_name
from objax.variable import VarCollection


class PrivateGradValues(Module):
    """Computes differentially private gradients as required by DP-SGD.
    This module can be used in place of GradVals, and automatically makes the optimizer differentially private."""

    def __init__(self,
                 f: Callable,
                 vc: VarCollection,
                 noise_multiplier: float,
                 l2_norm_clip: float,
                 microbatch: int,
                 batch_axis: Tuple[Optional[int], ...] = (0,),
                 keygen: random.Generator = random.DEFAULT_GENERATOR,
                 use_norm_accumulation: bool = False):
        """Constructs a PrivateGradValues instance.

        Args:
            f: the function for which to compute gradients.
            vc: the variables for which to compute gradients.
            noise_multiplier: scale of standard deviation for added noise in DP-SGD.
            l2_norm_clip: value of clipping norm for DP-SGD.
            microbatch: the size of each microbatch.
            batch_axis: the axis to use as batch during vectorization. Should be a tuple of 0s.
            keygen: a Generator for random numbers. Defaults to objax.random.DEFAULT_GENERATOR.
            use_norm_accumulation: whether to use norm accumulation as a first step to compute the clipped
              gradients (more details further). This is more memory efficient and often faster when the model
              is very deep and uses a large batch-size.
        """
        super().__init__()

        if not all(v == 0 for v in batch_axis):
            raise ValueError('batch_axis needs to be an all zero tuple for PrivateGradValues.')

        self.__wrapped__ = gv = GradValues(f, vc)

        self.batch_axis = batch_axis
        self.microbatch = microbatch
        self.l2_norm_clip = l2_norm_clip
        self.noise_multiplier = noise_multiplier
        self.keygen = keygen

        if use_norm_accumulation:
            self.clipped_grad = self._make_clipped_grad_fn_norm_accumulation(f, gv, vc)
        else:
            self.clipped_grad = self._make_clipped_grad_fn_simple(f, gv, vc)

    def _make_clipped_grad_fn_simple(self, f: Callable, gv: GradValues, vc: VarCollection) -> Callable:
        """Creates a function that computes gradients clipped per-sample.
        This algorithm vectorizes the computation of a clipped gradient defined over a single sample.

        Args:
          f: the function for which to compute gradients.
          gv: the GradValues object that computes non-clipped gradients.
          vc: the variables for which to compute gradients.

        Returns:
          clipped_grad: the function computing the average of gradients clipped per-sample.
        """
        @Function.with_vars(gv.vars())
        def clipped_grad_single_example(*args):
            grads, values = gv(*args)
            total_grad_norm = jn.linalg.norm([jn.linalg.norm(g) for g in grads])
            idivisor = 1 / jn.maximum(total_grad_norm / self.l2_norm_clip, 1.)
            return [g * idivisor for g in grads], values

        clipped_grad_vectorized = Vectorize(clipped_grad_single_example,
                                            batch_axis=self.batch_axis)

        def clipped_grad(*args):
            g, v = clipped_grad_vectorized(*args)
            g, v = jax.tree_map(functools.partial(jn.mean, axis=0), (g, v))
            return g, v

        return clipped_grad

    def _make_clipped_grad_fn_norm_accumulation(self, f: Callable, gv: GradValues, vc: VarCollection) -> Callable:
        """Creates a function that computes gradients clipped per-sample.
        This algorithm first accumulates the norm of the gradient per sample, and then
        performs an additional weighted backward pass that is equivalent to per-sample clipping.
        The accumulation allows jax to free up memory whenever the automatic differentiation engine
        is done processing a layer. This is more memory efficient and can be faster than the "simple" approach
        defined above, in particular at large batch-sizes for deep models.

        Reference: https://arxiv.org/abs/2009.03106.

        Args:
          f: the function for which to compute gradients.
          gv: the GradValues object that computes non-clipped gradients.
          vc: the variables for which to compute gradients.

        Returns:
          clipped_grad: the function computing the average of gradients clipped per-sample.
        """
        @Function.with_vars(gv.vars())
        def grad_norm_fn(*args):
            grads, values = gv(*args)
            total_grad_norm = jn.linalg.norm([jn.linalg.norm(g) for g in grads])
            return total_grad_norm, values

        grad_norm_fn_vectorized = Vectorize(grad_norm_fn, batch_axis=self.batch_axis)
        loss_fn_vectorized = Vectorize(Function.with_vars(gv.vars())(f), batch_axis=self.batch_axis)

        def weighted_loss_fn(weights, *args):
            returned = loss_fn_vectorized(*args)
            # the following assumes that the loss is the first element output by loss_fn_vectorized
            if isinstance(returned, jn.ndarray):
                loss_vector = returned
            else:
                loss_vector = returned[0]
            return jn.mean(loss_vector * weights)

        weighted_grad_fn = GradValues(weighted_loss_fn, vc)

        def clipped_grad(*args):
            grad_norms, values = grad_norm_fn_vectorized(*args)
            idivisor = 1 / jn.maximum(grad_norms / self.l2_norm_clip, 1.)
            clipped_grads, _ = weighted_grad_fn(idivisor, *args)
            return clipped_grads, jax.tree_map(functools.partial(jn.mean, axis=0), values)

        return clipped_grad

    def reshape_microbatch(self, x: JaxArray) -> JaxArray:
        """Reshapes examples into microbatches.
        DP-SGD requires that per-example gradients are clipped and noised, however this can be inefficient.
        To speed this up, it is possible to clip and noise a microbatch of examples, at a sight cost to privacy.
        If speed is not an issue, the microbatch size should be set to 1.

        If x has shape [D0, D1, ..., Dn], the reshaped output will
        have shape [number_of_microbatches, microbatch_size, D1, ..., DN].

        Args:
            x: items to be reshaped.

        Returns:
            The reshaped items.
        """
        s = x.shape
        return x.reshape([s[0] // self.microbatch, self.microbatch, *s[1:]])

    def __call__(self, *args):
        """Returns the computed DP-SGD gradients.

        Returns:
            A tuple (gradients, value of f)."""
        batch = args[0].shape[0]
        assert batch % self.microbatch == 0
        num_microbatches = batch // self.microbatch
        stddev = self.l2_norm_clip * self.noise_multiplier / num_microbatches
        g, v = self.clipped_grad(*[self.reshape_microbatch(x) for x in args])
        g = [gx + random.normal(gx.shape, stddev=stddev, generator=self.keygen) for gx in g]
        return g, v

    def __repr__(self):
        args = dict(f=repr_function(self.__wrapped__.f), noise_multiplier=self.noise_multiplier,
                    l2_norm_clip=self.l2_norm_clip, microbatch=self.microbatch, batch_axis=self.batch_axis)
        args = ', '.join(f'{k}={v}' for k, v in args.items())
        return f'{class_name(self)}({args})'
