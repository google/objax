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

from typing import List

from objax.module import Module
from objax.typing import JaxArray

try:
    # Only import tensorflow if available.
    import tensorflow as tf

    tf.config.experimental.set_visible_devices([], 'GPU')
except ImportError:
    # Make fake tf, so code in this file will be successfully imported even when Tensorflow is not installed.
    tf = type('tf', (), {})
    setattr(tf, 'Module', object)

    def _fake_tf_function(func=None, **kwargs):
        del kwargs
        if func is not None:
            return func
        else:
            return lambda x: x

    setattr(tf, 'function', _fake_tf_function)


class Objax2Tf(tf.Module):
    """Objax to Tensorflow converter, which converts Objax module to tf.Module."""

    def __init__(self, module: Module):
        """Create a Tensorflow module from Objax module.

        Args:
            module: Objax module to be converted to Tensorflow tf.Module.
        """
        from jax.experimental import jax2tf
        assert hasattr(tf, '__version__'), 'Tensorflow must be installed for Objax2Tf to work.'
        assert tf.__version__ >= '2.0', 'Objax2Tf works only with Tensorflow 2.'
        assert isinstance(module, Module), 'Input argument to Objax2Tf must be an Objax module.'

        super().__init__()

        module_vars = module.vars()

        def wrapped_op(tensor_list: List[JaxArray], kwargs, *args):
            original_values = module_vars.tensors()
            try:
                module_vars.assign(tensor_list)
                return module(*args, **kwargs)
            finally:
                module_vars.assign(original_values)

        tf_function = jax2tf.convert(wrapped_op)
        self._tf_vars = [tf.Variable(v) for v in module_vars.tensors()]
        self._tf_call = tf_function

    @tf.function(autograph=False)
    def __call__(self, *args, **kwargs):
        """Calls Tensorflow function which was generated from Objax module."""
        return self._tf_call(self._tf_vars, kwargs, *args)
