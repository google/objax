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
from tqdm import trange

import objax
from objax.util import EasyDict


def simple_net_block(nin, nout):
    return objax.nn.Sequential([
        objax.nn.Conv2D(nin, nout, k=3), objax.functional.leaky_relu,
        objax.functional.max_pool_2d,
        objax.nn.Conv2D(nout, nout, k=3), objax.functional.leaky_relu,
    ])


class SimpleNet(objax.Module):
    def __init__(self, nclass, colors, n):
        self.pre_conv = objax.nn.Sequential([objax.nn.Conv2D(colors, n, k=3), objax.functional.leaky_relu])
        self.block1 = simple_net_block(1 * n, 2 * n)
        self.block2 = simple_net_block(2 * n, 4 * n)
        self.post_conv = objax.nn.Conv2D(4 * n, nclass, k=3)

    def __call__(self, x, training=False):  # x = (batch, colors, height, width)
        y = self.pre_conv(x)
        y = self.block1(y)
        y = self.block2(y)
        logits = self.post_conv(y).mean((2, 3))  # logits = (batch, nclass)
        if training:
            return logits
        return objax.functional.softmax(logits)


# Data
DATA_DIR = os.path.join(os.environ['HOME'], 'TFDS')
data = tfds.as_numpy(tfds.load(name='mnist', batch_size=-1, data_dir=DATA_DIR))
train = EasyDict(image=data['train']['image'].transpose(0, 3, 1, 2) / 255, label=data['train']['label'])
test = EasyDict(image=data['test']['image'].transpose(0, 3, 1, 2) / 255, label=data['test']['label'])
del data


def augment(x, shift=4):  # Shift all images in the batch by up to "shift" pixels in any direction.
    x_pad = np.pad(x, [[0, 0], [0, 0], [shift, shift], [shift, shift]])
    rx, ry = np.random.randint(0, shift, size=2)
    return x_pad[:, :, rx:rx + 28, ry:ry + 28]


# Settings
batch = 512
test_batch = 2048
weight_decay = 0.0001
epochs = 40
lr = 0.0004 * (batch / 64)
train_size = train.image.shape[0]

# Model
model = SimpleNet(nclass=10, colors=1, n=16)  # Use higher values of n to get higher accuracy.
opt = objax.optimizer.Adam(model.vars())
ema = objax.optimizer.ExponentialMovingAverage(model.vars(), momentum=0.999, debias=True)
predict = objax.Jit(ema.replace_vars(model), model.vars() + ema.vars())


def loss(x, y):
    logits = model(x, training=True)
    loss_xe = objax.functional.loss.cross_entropy_logits_sparse(logits, y).mean()
    loss_l2 = 0.5 * sum((v.value ** 2).sum() for k, v in model.vars().items() if k.endswith('.w'))
    return loss_xe + weight_decay * loss_l2, {'loss/xe': loss_xe, 'loss/l2': loss_l2}


gv = objax.GradValues(loss, model.vars())


def train_op(x, y):
    g, v = gv(x, y)
    opt(lr, g)
    ema()
    return v


# gv.vars() contains the model variables.
train_op = objax.Jit(train_op, gv.vars() + opt.vars() + ema.vars())

# Training
model.vars().print()
for epoch in range(epochs):
    # Train one epoch
    loop = trange(0, train_size, batch,
                  leave=False, unit='img', unit_scale=batch,
                  desc='Epoch %d/%d ' % (1 + epoch, epochs))
    for it in loop:
        sel = np.random.randint(size=(batch,), low=0, high=train.image.shape[0])
        v = train_op(augment(train.image[sel]), train.label[sel])

    # Eval
    accuracy = 0
    for it in trange(0, test.image.shape[0], test_batch, leave=False, desc='Evaluating'):
        x = test.image[it: it + test_batch]
        xl = test.label[it: it + test_batch]
        accuracy += (np.argmax(predict(x), axis=1) == xl).sum()
    accuracy /= test.image.shape[0]
    print(f'Epoch {epoch + 1:04d}  Accuracy {100 * accuracy:.2f}')
