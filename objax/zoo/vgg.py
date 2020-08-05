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

"""Module with VGG-19 implementation.

See https://arxiv.org/abs/1409.1556 for detail.
"""

import functools
import os
from urllib import request

import jax.numpy as jn
import numpy as np

import objax

_VGG19_URL = 'https://github.com/machrisaa/tensorflow-vgg'
_VGG19_NPY = './objax/zoo/pretrained/vgg19.npy'
_SYNSET_URL = 'https://raw.githubusercontent.com/machrisaa/tensorflow-vgg/master/synset.txt'
_SYNSET_PATH = './objax/zoo/pretrained/synset.txt'


def preprocess(x):
    bgr_mean = [103.939, 116.779, 123.68]
    red, green, blue = [x[:, i, :, :] for i in range(3)]
    return jn.stack([blue - bgr_mean[0], green - bgr_mean[1], red - bgr_mean[2]], axis=1)


def max_pool_2d(x):
    return functools.partial(objax.functional.max_pool_2d,
                             size=2, strides=2, padding=objax.constants.ConvPadding.VALID)(x)


class VGG19(objax.nn.Sequential):
    """VGG19 implementation."""

    def __init__(self, pretrained=False):
        """Creates VGG19 instance.

        Args:
            pretrained: if True load weights from ImageNet pretrained model.
        """
        if not os.path.exists(_VGG19_NPY):
            raise FileNotFoundError(
                'You must download vgg19.npy from %s and save it to %s' % (_VGG19_URL, _VGG19_NPY))
        if not os.path.exists(_SYNSET_PATH):
            request.urlretrieve(_SYNSET_URL, _SYNSET_PATH)
        self.data_dict = np.load(_VGG19_NPY, encoding='latin1', allow_pickle=True).item()
        self.pretrained = pretrained
        self.ops = self.build()
        super().__init__(self.ops)

    def build(self):
        # inputs in [0, 255]
        self.preprocess = preprocess
        self.conv1_1 = objax.nn.Conv2D(nin=3, nout=64, k=3)
        self.relu1_1 = objax.functional.relu
        self.conv1_2 = objax.nn.Conv2D(nin=64, nout=64, k=3)
        self.relu1_2 = objax.functional.relu
        self.pool1 = max_pool_2d

        self.conv2_1 = objax.nn.Conv2D(nin=64, nout=128, k=3)
        self.relu2_1 = objax.functional.relu
        self.conv2_2 = objax.nn.Conv2D(nin=128, nout=128, k=3)
        self.relu2_2 = objax.functional.relu
        self.pool2 = max_pool_2d

        self.conv3_1 = objax.nn.Conv2D(nin=128, nout=256, k=3)
        self.relu3_1 = objax.functional.relu
        self.conv3_2 = objax.nn.Conv2D(nin=256, nout=256, k=3)
        self.relu3_2 = objax.functional.relu
        self.conv3_3 = objax.nn.Conv2D(nin=256, nout=256, k=3)
        self.relu3_3 = objax.functional.relu
        self.conv3_4 = objax.nn.Conv2D(nin=256, nout=256, k=3)
        self.relu3_4 = objax.functional.relu
        self.pool3 = max_pool_2d

        self.conv4_1 = objax.nn.Conv2D(nin=256, nout=512, k=3)
        self.relu4_1 = objax.functional.relu
        self.conv4_2 = objax.nn.Conv2D(nin=512, nout=512, k=3)
        self.relu4_2 = objax.functional.relu
        self.conv4_3 = objax.nn.Conv2D(nin=512, nout=512, k=3)
        self.relu4_3 = objax.functional.relu
        self.conv4_4 = objax.nn.Conv2D(nin=512, nout=512, k=3)
        self.relu4_4 = objax.functional.relu
        self.pool4 = max_pool_2d

        self.conv5_1 = objax.nn.Conv2D(nin=512, nout=512, k=3)
        self.relu5_1 = objax.functional.relu
        self.conv5_2 = objax.nn.Conv2D(nin=512, nout=512, k=3)
        self.relu5_2 = objax.functional.relu
        self.conv5_3 = objax.nn.Conv2D(nin=512, nout=512, k=3)
        self.relu5_3 = objax.functional.relu
        self.conv5_4 = objax.nn.Conv2D(nin=512, nout=512, k=3)
        self.relu5_4 = objax.functional.relu
        self.pool5 = max_pool_2d

        self.flatten = objax.functional.flatten
        self.fc6 = objax.nn.Linear(nin=512 * 7 * 7, nout=4096)
        self.relu6 = objax.functional.relu
        self.fc7 = objax.nn.Linear(nin=4096, nout=4096)
        self.relu7 = objax.functional.relu
        self.fc8 = objax.nn.Linear(nin=4096, nout=1000)

        if self.pretrained:
            for it in self.data_dict:
                if it.startswith('conv'):
                    conv = getattr(self, it)
                    kernel, bias = self.data_dict[it]
                    conv.w = objax.TrainVar(jn.array(kernel))
                    conv.b = objax.TrainVar(jn.array(bias[:, None, None]))
                    setattr(self, it, conv)
                elif it.startswith('fc'):
                    linear = getattr(self, it)
                    kernel, bias = self.data_dict[it]
                    if it == 'fc6':
                        kernel = kernel.reshape([7, 7, 512, -1]).transpose((2, 0, 1, 3)).reshape([512 * 7 * 7, -1])
                    linear.w = objax.TrainVar(jn.array(kernel))
                    linear.b = objax.TrainVar(jn.array(bias))
                    setattr(self, it, linear)

        ops = [self.conv1_1, self.relu1_1, self.conv1_2, self.relu1_2, self.pool1,
               self.conv2_1, self.relu2_1, self.conv2_2, self.relu2_2, self.pool2,
               self.conv3_1, self.relu3_1, self.conv3_2, self.relu3_2,
               self.conv3_3, self.relu3_3, self.conv3_4, self.relu3_4, self.pool3,
               self.conv4_1, self.relu4_1, self.conv4_2, self.relu4_2,
               self.conv4_3, self.relu4_3, self.conv4_4, self.relu4_4, self.pool4,
               self.conv5_1, self.relu5_1, self.conv5_2, self.relu5_2,
               self.conv5_3, self.relu5_3, self.conv5_4, self.relu5_4, self.pool5,
               self.flatten, self.fc6, self.relu6, self.fc7, self.relu7, self.fc8]

        return ops
