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

__all__ = ['assert_assigned_type_and_shape_match']

import jax

from objax.typing import JaxArray


TRACER_TYPES = (jax.interpreters.partial_eval.JaxprTracer,
                jax.interpreters.partial_eval.DynamicJaxprTracer)


def split_shape_and_device(array):
    if isinstance(array, jax.interpreters.pxla.ShardedDeviceArray):
        return array.shape[0], array.shape[1:]
    else:
        return None, array.shape


def assert_assigned_type_and_shape_match(existing_tensor, new_tensor):
    assert isinstance(new_tensor, JaxArray.__args__), \
        f'Assignments to variable must be an instance of JaxArray, but received f{type(new_tensor)}.'

    new_tensor_device, new_tensor_shape = split_shape_and_device(new_tensor)
    self_device, self_shape = split_shape_and_device(existing_tensor)

    device_mismatch_error = f'Can not replicate a variable that is currently on ' \
                            f'{self_device} devices to {new_tensor_device} devices.'
    assert (new_tensor_device is None) or (self_device is None) or (self_device == new_tensor_device), \
        device_mismatch_error

    shorter_length = min(len(new_tensor.shape), len(existing_tensor.shape))
    is_special_ok = (isinstance(new_tensor, TRACER_TYPES) or isinstance(existing_tensor, TRACER_TYPES))
    is_special_ok = is_special_ok and existing_tensor.shape[-shorter_length:] == new_tensor.shape[-shorter_length:]

    shape_mismatch_error = f'Assign can not change shape of variable. The current variable shape is {self_shape},' \
                           f' but the requested new shape is {new_tensor_shape}.'
    assert is_special_ok or new_tensor_shape == self_shape or new_tensor.shape == existing_tensor.shape, \
        shape_mismatch_error
