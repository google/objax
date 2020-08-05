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

import os

import numpy as np
import tensorflow_datasets as tfds

import objax
from objax.util import EasyDict

# Data: train has 1027 images - test has 256 images
# Each image is 300 x 300 x 3 bytes
DATA_DIR = os.path.join(os.environ['HOME'], 'TFDS')
data = tfds.as_numpy(tfds.load(name='horses_or_humans', batch_size=-1, data_dir=DATA_DIR))


def prepare(x, downscale=3):
    """Normalize images to [-1, 1] and downscale them to 100x100x3 (for faster training) and flatten them."""
    s = x.shape
    x = x.astype('f').reshape((s[0], s[1] // downscale, downscale, s[2] // downscale, downscale, s[3]))
    return x.mean((2, 4)).reshape((s[0], -1)) * (1 / 127.5) - 1


train = EasyDict(image=prepare(data['train']['image']), label=data['train']['label'])
test = EasyDict(image=prepare(data['test']['image']), label=data['test']['label'])
ndim = train.image.shape[-1]
del data

# Settings
lr = 0.0001  # learning rate
batch = 256
epochs = 20

# Model
model = objax.nn.Linear(ndim, 1)
opt = objax.optimizer.SGD(model.vars())
model.vars().print()


# Cross Entropy Loss
def loss(x, label):
    return objax.functional.loss.sigmoid_cross_entropy_logits(model(x)[:, 0], label).mean()


gv = objax.GradValues(loss, model.vars())


def train_op(x, label):
    g, v = gv(x, label)  # returns gradients, loss
    opt(lr, g)
    return v


# This line is optional: it is compiling the code to make it faster.
# gv.vars() contains the model variables.
train_op = objax.Jit(train_op, gv.vars() + opt.vars())

# Training
for epoch in range(epochs):
    # Train
    avg_loss = 0
    for it in range(0, train.image.shape[0], batch):
        sel = np.random.randint(size=(batch,), low=0, high=train.image.shape[0])
        avg_loss += float(train_op(train.image[sel], train.label[sel])[0]) * batch
    avg_loss /= it + batch

    # Eval
    accuracy = 0
    for it in range(0, test.image.shape[0], batch):
        x, y = test.image[it: it + batch], test.label[it: it + batch]
        accuracy += (np.round(objax.functional.sigmoid(model(x)))[:, 0] == y).sum()
    accuracy /= test.image.shape[0]
    print('Epoch %04d  Loss %.2f  Accuracy %.2f' % (epoch + 1, avg_loss, 100 * accuracy))
