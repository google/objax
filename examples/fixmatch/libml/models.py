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

from examples.fixmatch.libml.zoo.convnet import ConvNet
from examples.fixmatch.libml.zoo.resnet import ResNet

ARCHS = 'convnet resnet'.split()


def network(arch: str):
    if arch == 'convnet':
        return ConvNet
    elif arch == 'resnet':
        return ResNet
    raise ValueError('Architecture not recognized', arch)
