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

"""Module with ResNet-v2 implementation.

See https://arxiv.org/abs/1603.05027 for detail.
"""

import functools
from typing import Callable, Sequence, Union, Optional

from jax import numpy as jn

import objax
from objax.constants import ConvPadding
from objax.nn import Conv2D
from objax.typing import JaxArray, ConvPaddingInt

try:
    # Imports tensorflow when loading model weights from pretrained keras model.
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')
except ImportError:
    tf = None

__all__ = ['ResNetV2', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'ResNet200',
           'load_pretrained_weights_from_keras']


def conv_args(kernel_size: int,
              nout: int,
              padding: Optional[Union[ConvPadding, str, ConvPaddingInt]] = ConvPadding.VALID):
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
                padding=padding)


class ResNetV2Block(objax.Module):
    """ResNet v2 block with optional bottleneck."""

    def __init__(self,
                 nin: int,
                 nout: int,
                 stride: Union[int, Sequence[int]],
                 use_projection: bool,
                 bottleneck: bool,
                 normalization_fn: Callable[..., objax.Module] = objax.nn.BatchNorm2D,
                 activation_fn: Callable[[JaxArray], JaxArray] = objax.functional.relu):
        """Creates ResNetV2Block instance.

        Args:
            nin: number of input filters.
            nout: number of output filters.
            stride: stride for 3x3 convolution and projection convolution in this block.
            use_projection: if True then include projection convolution into this block.
            bottleneck: if True then make bottleneck block.
            normalization_fn: module which used as normalization function.
            activation_fn: activation function.
        """
        self.use_projection = use_projection
        self.activation_fn = activation_fn
        self.stride = stride

        if self.use_projection:
            self.proj_conv = Conv2D(nin, nout, 1, strides=stride, **conv_args(1, nout))

        if bottleneck:
            self.norm_0 = normalization_fn(nin)
            self.conv_0 = Conv2D(nin, nout // 4, 1, strides=1, **conv_args(1, nout // 4))
            self.norm_1 = normalization_fn(nout // 4)
            self.conv_1 = Conv2D(nout // 4, nout // 4, 3, strides=stride, **conv_args(3, nout // 4, (1, 1)))
            self.norm_2 = normalization_fn(nout // 4)
            self.conv_2 = Conv2D(nout // 4, nout, 1, strides=1, **conv_args(1, nout))
            self.layers = ((self.norm_0, self.conv_0), (self.norm_1, self.conv_1), (self.norm_2, self.conv_2))
        else:
            self.norm_0 = normalization_fn(nin)
            self.conv_0 = Conv2D(nin, nout, 3, strides=1, **conv_args(3, nout, (1, 1)))
            self.norm_1 = normalization_fn(nout)
            self.conv_1 = Conv2D(nout, nout, 3, strides=stride, **conv_args(3, nout, (1, 1)))
            self.layers = ((self.norm_0, self.conv_0), (self.norm_1, self.conv_1))

    def __call__(self, x: JaxArray, training: bool) -> JaxArray:
        if self.stride > 1:
            shortcut = objax.functional.max_pool_2d(x, size=1, strides=self.stride)
        else:
            shortcut = x

        for i, (bn_i, conv_i) in enumerate(self.layers):
            x = bn_i(x, training)
            x = self.activation_fn(x)
            if i == 0 and self.use_projection:
                shortcut = self.proj_conv(x)
            x = conv_i(x)

        return x + shortcut


class ResNetV2BlockGroup(objax.nn.Sequential):
    """Group of ResNet v2 Blocks."""

    def __init__(self,
                 nin: int,
                 nout: int,
                 num_blocks: int,
                 stride: Union[int, Sequence[int]],
                 use_projection: bool,
                 bottleneck: bool,
                 normalization_fn: Callable[..., objax.Module] = objax.nn.BatchNorm2D,
                 activation_fn: Callable[[JaxArray], JaxArray] = objax.functional.relu):
        """Creates ResNetV2BlockGroup instance.

        Args:
            nin: number of input filters.
            nout: number of output filters.
            num_blocks: number of Resnet blocks in this group.
            stride: stride for 3x3 convolutions and projection convolutions in Resnet blocks.
            use_projection: if True then include projection convolution into each Resnet blocks.
            bottleneck: if True then make bottleneck blocks.
            normalization_fn: module which used as normalization function.
            activation_fn: activation function.
        """
        blocks = []
        for i in range(num_blocks):
            blocks.append(
                ResNetV2Block(
                    nin=(nin if i == 0 else nout),
                    nout=nout,
                    stride=(stride if i == num_blocks - 1 else 1),
                    use_projection=(i == 0 and use_projection),
                    bottleneck=bottleneck,
                    normalization_fn=normalization_fn,
                    activation_fn=activation_fn))
        super().__init__(blocks)


class ResNetV2(objax.nn.Sequential):
    """Base implementation of ResNet v2 from https://arxiv.org/abs/1603.05027."""

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 blocks_per_group: Sequence[int],
                 bottleneck: bool = True,
                 channels_per_group: Sequence[int] = (256, 512, 1024, 2048),
                 group_strides: Sequence[int] = (1, 2, 2, 2),
                 group_use_projection: Sequence[bool] = (True, True, True, True),
                 normalization_fn: Callable[..., objax.Module] = objax.nn.BatchNorm2D,
                 activation_fn: Callable[[JaxArray], JaxArray] = objax.functional.relu):
        """Creates ResNetV2 instance.

        Args:
            in_channels: number of channels in the input image.
            num_classes: number of output classes.
            blocks_per_group: number of blocks in each block group.
            bottleneck: if True then use bottleneck blocks.
            channels_per_group: number of output channels for each block group.
            group_strides: strides for each block group.
            normalization_fn: module which used as normalization function.
            activation_fn: activation function.
        """
        assert len(channels_per_group) == len(blocks_per_group)
        assert len(group_strides) == len(blocks_per_group)
        assert len(group_use_projection) == len(blocks_per_group)
        nin = in_channels
        nout = 64
        ops = [Conv2D(nin, nout, k=7, strides=2, **conv_args(7, 64, (3, 3))),
               functools.partial(jn.pad, pad_width=((0, 0), (0, 0), (1, 1), (1, 1))),
               functools.partial(objax.functional.max_pool_2d,
                                 size=3, strides=2, padding=ConvPadding.VALID)]
        for i in range(len(blocks_per_group)):
            nin = nout
            nout = channels_per_group[i]
            ops.append(ResNetV2BlockGroup(
                nin,
                nout,
                num_blocks=blocks_per_group[i],
                stride=group_strides[i],
                bottleneck=bottleneck,
                use_projection=group_use_projection[i],
                normalization_fn=normalization_fn,
                activation_fn=activation_fn))

        ops.extend([normalization_fn(nout),
                    activation_fn,
                    lambda x: x.mean((2, 3)),
                    objax.nn.Linear(nout,
                                    num_classes,
                                    w_init=objax.nn.init.xavier_normal)])
        super().__init__(ops)


class ResNet18(ResNetV2):
    """Implementation of ResNet v2 with 18 layers."""

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 normalization_fn: Callable[..., objax.Module] = objax.nn.BatchNorm2D,
                 activation_fn: Callable[[JaxArray], JaxArray] = objax.functional.relu):
        """Creates ResNet18 instance.

        Args:
            in_channels: number of channels in the input image.
            num_classes: number of output classes.
            normalization_fn: module which used as normalization function.
            activation_fn: activation function.
        """
        super().__init__(in_channels=in_channels,
                         num_classes=num_classes,
                         blocks_per_group=(2, 2, 2, 2),
                         bottleneck=False,
                         channels_per_group=(64, 128, 256, 512),
                         group_use_projection=(False, True, True, True),
                         normalization_fn=normalization_fn,
                         activation_fn=activation_fn)


class ResNet34(ResNetV2):
    """Implementation of ResNet v2 with 34 layers."""

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 normalization_fn: Callable[..., objax.Module] = objax.nn.BatchNorm2D,
                 activation_fn: Callable[[JaxArray], JaxArray] = objax.functional.relu):
        """Creates ResNet34 instance.

        Args:
            in_channels: number of channels in the input image.
            num_classes: number of output classes.
            normalization_fn: module which used as normalization function.
            activation_fn: activation function.
        """
        super().__init__(in_channels=in_channels,
                         num_classes=num_classes,
                         blocks_per_group=(3, 4, 6, 3),
                         bottleneck=False,
                         channels_per_group=(64, 128, 256, 512),
                         group_use_projection=(False, True, True, True),
                         normalization_fn=normalization_fn,
                         activation_fn=activation_fn)


class ResNet50(ResNetV2):
    """Implementation of ResNet v2 with 50 layers."""

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 normalization_fn: Callable[..., objax.Module] = objax.nn.BatchNorm2D,
                 activation_fn: Callable[[JaxArray], JaxArray] = objax.functional.relu):
        """Creates ResNet50 instance.

        Args:
            in_channels: number of channels in the input image.
            num_classes: number of output classes.
            normalization_fn: module which used as normalization function.
            activation_fn: activation function.
        """
        super().__init__(in_channels=in_channels,
                         num_classes=num_classes,
                         blocks_per_group=(3, 4, 6, 3),
                         group_strides=(2, 2, 2, 1),
                         bottleneck=True,
                         normalization_fn=normalization_fn,
                         activation_fn=activation_fn)


class ResNet101(ResNetV2):
    """Implementation of ResNet v2 with 101 layers."""

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 normalization_fn: Callable[..., objax.Module] = objax.nn.BatchNorm2D,
                 activation_fn: Callable[[JaxArray], JaxArray] = objax.functional.relu):
        """Creates ResNet101 instance.

        Args:
            in_channels: number of channels in the input image.
            num_classes: number of output classes.
            normalization_fn: module which used as normalization function.
            activation_fn: activation function.
        """
        super().__init__(in_channels=in_channels,
                         num_classes=num_classes,
                         blocks_per_group=(3, 4, 23, 3),
                         group_strides=(2, 2, 2, 1),
                         bottleneck=True,
                         normalization_fn=normalization_fn,
                         activation_fn=activation_fn)


class ResNet152(ResNetV2):
    """Implementation of ResNet v2 with 152 layers."""

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 normalization_fn: Callable[..., objax.Module] = objax.nn.BatchNorm2D,
                 activation_fn: Callable[[JaxArray], JaxArray] = objax.functional.relu):
        """Creates ResNet152 instance.

        Args:
            in_channels: number of channels in the input image.
            num_classes: number of output classes.
            normalization_fn: module which used as normalization function.
            activation_fn: activation function.
        """
        super().__init__(in_channels=in_channels,
                         num_classes=num_classes,
                         blocks_per_group=(3, 8, 36, 3),
                         group_strides=(2, 2, 2, 1),
                         bottleneck=True,
                         normalization_fn=normalization_fn,
                         activation_fn=activation_fn)


class ResNet200(ResNetV2):
    """Implementation of ResNet v2 with 200 layers."""

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 normalization_fn: Callable[..., objax.Module] = objax.nn.BatchNorm2D,
                 activation_fn: Callable[[JaxArray], JaxArray] = objax.functional.relu):
        """Creates ResNet200 instance.

        Args:
            in_channels: number of channels in the input image.
            num_classes: number of output classes.
            normalization_fn: module which used as normalization function.
            activation_fn: activation function.
        """
        super().__init__(in_channels=in_channels,
                         num_classes=num_classes,
                         blocks_per_group=(3, 24, 36, 3),
                         group_strides=(2, 2, 2, 1),
                         bottleneck=True,
                         normalization_fn=normalization_fn,
                         activation_fn=activation_fn)


def convert_bn_layer(keras_layer, objax_layer):
    """Converts variables of batch normalization layer from Keras to Objax."""
    shape = objax_layer.gamma.value.shape
    objax_layer.gamma = objax.TrainVar(jn.array(keras_layer.__dict__['gamma'].numpy()).reshape(shape))
    objax_layer.beta = objax.TrainVar(jn.array(keras_layer.__dict__['beta'].numpy()).reshape(shape))
    objax_layer.running_mean = objax.StateVar(jn.array(keras_layer.__dict__['moving_mean'].numpy()).reshape(shape))
    objax_layer.running_var = objax.StateVar(jn.array(keras_layer.__dict__['moving_variance'].numpy()).reshape(shape))


def convert_conv_or_linear_layer(keras_layer, objax_layer):
    """Converts variables of convolution/linear layer from Keras to Objax."""
    objax_layer.w = objax.TrainVar(jn.array(keras_layer.__dict__['kernel'].numpy()))
    if keras_layer.__dict__.get('bias') is not None:
        if jn.ndim(objax_layer.w.value) == 4:
            # conv layer
            bias_shape = (objax_layer.w.value.shape[-1], 1, 1)
        else:
            # linear layer
            bias_shape = (objax_layer.w.value.shape[-1])
        objax_layer.b = objax.TrainVar(jn.array(keras_layer.__dict__['bias'].numpy()).reshape(bias_shape))
    else:
        objax_layer.b = None


def convert_resblock(keras_block, objax_block):
    """Converts variables of residual block from Keras to Objax.

    Name mapping between blocks:
        proj_conv   : 0_conv (when exists)
        conv_0      : 1_conv
        conv_1      : 2_conv
        conv_2      : 3_conv
        norm_0      : preact_bn
        norm_1      : 1_bn
        norm_2      : 2_bn
    """
    def _find_layer(suffix):
        """Finds a layer whose name contains a suffix."""
        return next(layer for layer in keras_block if layer.name.endswith(suffix))

    if hasattr(objax_block, 'proj_conv'):
        convert_conv_or_linear_layer(_find_layer('0_conv'), objax_block.proj_conv)
    convert_conv_or_linear_layer(_find_layer('1_conv'), objax_block.conv_0)
    convert_conv_or_linear_layer(_find_layer('2_conv'), objax_block.conv_1)
    convert_conv_or_linear_layer(_find_layer('3_conv'), objax_block.conv_2)
    convert_bn_layer(_find_layer('preact_bn'), objax_block.norm_0)
    convert_bn_layer(_find_layer('1_bn'), objax_block.norm_1)
    convert_bn_layer(_find_layer('2_bn'), objax_block.norm_2)


def convert_keras_model(model_keras, model_objax, num_blocks: list, include_top: bool = True):
    """Converts Keras implementation of ResNetV2 into Objax."""
    convert_conv_or_linear_layer(model_keras.get_layer('conv1_conv'), model_objax[0])
    for b, j in enumerate(num_blocks):
        for i in range(j):
            convert_resblock([layer for layer in model_keras.layers
                              if layer.name.startswith('conv{}_block{}'.format(b + 2, i + 1))],
                             model_objax[b + 3][i])
    convert_bn_layer(model_keras.get_layer('post_bn'), model_objax[7])
    if include_top:
        convert_conv_or_linear_layer(model_keras.get_layer('predictions'), model_objax[10])


def load_pretrained_weights_from_keras(arch: str, include_top: bool = True, num_classes: int = 1000):
    """Function to load weights from Keras models.

    Args:
        arch: network architecture in ['ResNet50', 'ResNet101', 'ResNet152'].
        include_top: if True then include top linear classification layer.
        num_classees: number of classes. If include_top is True, num_classes should be 1000.

    Returns:
        Objax ResNetV2 models with pretrained weights from Keras.
    """
    model_registry = {'ResNet50': {'num_blocks': [3, 4, 6, 3]},
                      'ResNet101': {'num_blocks': [3, 4, 23, 3]},
                      'ResNet152': {'num_blocks': [3, 8, 36, 3]}}
    assert tf is not None, 'Please install tensorflow dependency to be able to load keras weights.'
    assert arch in model_registry, f'Model weights does not exist for {arch}.'
    assert not include_top or num_classes == 1000, ('Number of classes should be 1000 when including top layer.')

    model_keras = getattr(tf.keras.applications, arch + 'V2')(include_top=include_top,
                                                              weights='imagenet',
                                                              classes=1000,
                                                              classifier_activation='linear')
    model_objax = objax.zoo.resnet_v2.__dict__[arch](in_channels=3,
                                                     num_classes=num_classes,
                                                     normalization_fn=functools.partial(objax.nn.BatchNorm2D,
                                                                                        momentum=0.99,
                                                                                        eps=1.001e-05))
    convert_keras_model(model_keras, model_objax, model_registry[arch]['num_blocks'], include_top)
    del model_keras
    return model_objax
