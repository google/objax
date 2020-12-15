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

import random

import numpy as np
import tensorflow as tf

import objax
from objax.zoo.wide_resnet import WideResNet

# Data
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()
X_train = X_train.transpose(0, 3, 1, 2) / 255.0
X_test = X_test.transpose(0, 3, 1, 2) / 255.0

# Model
model = WideResNet(nin=3, nclass=10, depth=28, width=2)
opt = objax.optimizer.Adam(model.vars())


# Losses
@objax.Function.with_vars(model.vars())
def loss(x, label):
    logit = model(x, training=True)
    return objax.functional.loss.cross_entropy_logits_sparse(logit, label).mean()


gv = objax.GradValues(loss, model.vars())


@objax.Function.with_vars(model.vars() + gv.vars() + opt.vars())
def train_op(x, y, lr):
    g, v = gv(x, y)
    opt(lr=lr, grads=g)
    return v


train_op = objax.Jit(train_op)
predict = objax.Jit(objax.nn.Sequential([
    objax.ForceArgs(model, training=False), objax.functional.softmax
]))


def augment(x):
    if random.random() < .5:
        x = x[:, :, :, ::-1]  # Flip the batch images about the horizontal axis
    # Pixel-shift all images in the batch by up to 4 pixels in any direction.
    x_pad = np.pad(x, [[0, 0], [0, 0], [4, 4], [4, 4]], 'reflect')
    rx, ry = np.random.randint(0, 4), np.random.randint(0, 4)
    x = x_pad[:, :, rx:rx + 32, ry:ry + 32]
    return x


# Training
print(model.vars())
for epoch in range(30):
    # Train
    loss = []
    sel = np.arange(len(X_train))
    np.random.shuffle(sel)
    for it in range(0, X_train.shape[0], 64):
        loss.append(train_op(augment(X_train[sel[it:it + 64]]), Y_train[sel[it:it + 64]].flatten(),
                             4e-3 if epoch < 20 else 4e-4))

    # Eval
    test_predictions = [predict(x_batch).argmax(1) for x_batch in X_test.reshape((50, -1) + X_test.shape[1:])]
    accuracy = np.array(test_predictions).flatten() == Y_test.flatten()
    print('Epoch %04d  Loss %.2f  Accuracy %.2f' % (epoch + 1, np.mean(loss), 100 * np.mean(accuracy)))
