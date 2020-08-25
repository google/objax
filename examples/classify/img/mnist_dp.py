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

import argparse
import os
from functools import partial

import numpy as np
import tensorflow_datasets as tfds
from tqdm import trange

import objax
from objax.functional import one_hot, tanh
from objax.functional.ops import average_pool_2d
from objax.jaxboard import SummaryWriter, Summary
from objax.nn import Sequential, Conv2D
from objax.util import EasyDict

parser = argparse.ArgumentParser()

parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--lr", type=float, default=0.25, help="Learning rate.")
parser.add_argument("--batchsize", type=int, default=256)

parser.add_argument("--l2_norm_clip", type=float, default=1, help="Clipping norm.")
parser.add_argument("--noise_multiplier", type=float, default=1.1, help="Noise multiplier.")
parser.add_argument("--microbatch", type=int, default=1, help="Microbatch size.")
parser.add_argument("--delta", type=float, default=1e-5, help="Target delta value for DP.")

parser.add_argument("--log_dir", type=str, default="e/mnist_dp")

args = parser.parse_args()


class ALLCNN(Sequential):
    def __init__(self, nin, nclass, scales, filters, filters_max):
        def nl(x):
            """Return tanh as activation function. Tanh has better utility for
            differentially private SGD https://arxiv.org/abs/2007.14191 .
            """
            return tanh(x)

        def nf(scale):
            return min(filters_max, filters << scale)

        ops = [Conv2D(nin, nf(0), 3), nl]
        for i in range(scales):
            ops.extend([Conv2D(nf(i), nf(i), 3), nl,
                        Conv2D(nf(i), nf(i + 1), 3), nl, partial(average_pool_2d, size=2, strides=2)])
        ops.extend([Conv2D(nf(scales), nclass, 3), lambda x: x.mean((2, 3))])
        super().__init__(ops)


# Data
DATA_DIR = os.path.join(os.environ['HOME'], 'TFDS')
data = tfds.as_numpy(tfds.load(name='mnist', batch_size=-1, data_dir=DATA_DIR))
inputs = data['train']['image']
labels = data['train']['label']
train = EasyDict(image=inputs.transpose(0, 3, 1, 2) / 127.5 - 1, label=labels)
test = EasyDict(image=data['test']['image'].transpose(0, 3, 1, 2) / 127.5 - 1,
                label=data['test']['label'])
num_train_images = train.image.shape[0]
nclass = len(np.unique(data['train']['label']))
del data, inputs, labels

# Settings
log_dir = args.log_dir
filters = 32
filters_max = 64

num_train_epochs = args.epochs
lr = args.lr
batch = args.batchsize
l2_norm_clip = args.l2_norm_clip
noise_multiplier = args.noise_multiplier
microbatch = args.microbatch
delta = args.delta

sampling_rate = batch / float(num_train_images)  # for privacy accounting.

# Model
model = ALLCNN(nin=1, nclass=nclass, scales=2, filters=filters, filters_max=filters_max)
model_vars = model.vars()
opt = objax.optimizer.SGD(model_vars)
predict = objax.Jit(model)
print(model_vars)


def loss(x, label):
    logit = model(x)
    return objax.functional.loss.cross_entropy_logits(logit, label).mean()


# This is the main part that is different from non-private training.
# We use PrivateGradValues instead of GradValues to get private gradient.
# batch_axis=(0, 0) indicates the axis to use as batch is 0. It should be set to
# an all 0 tuple for PrivateGradValues.
gv = objax.privacy.PrivateGradValues(loss, model_vars,
                                     noise_multiplier,
                                     l2_norm_clip,
                                     microbatch,
                                     batch_axis=(0, 0))


def train_op(x, xl):
    g, v = gv(x, xl)  # returns private gradients, loss
    opt(lr, g)
    return v


# gv.vars() contains model_vars.
# Different from GradValues, in the case of PrivateGradValues, gv.vars() has its
# own internal variable, the key of the random number generator.
# When we jit train_op, we need to have the interval variable passed to Jit.
train_op = objax.Jit(train_op, gv.vars() + opt.vars())

# Training
with SummaryWriter(os.path.join(log_dir, 'tb')) as tensorboard:
    steps = 0  # Keep track the number of iterations for privacy accounting.
    for epoch in range(num_train_epochs):
        # Train one epoch
        summary = Summary()
        loop = trange(0, num_train_images, batch,
                      leave=False, unit='img', unit_scale=batch,
                      desc='Epoch %d/%d' % (1 + epoch, num_train_epochs))
        for it in loop:
            sel = np.random.randint(size=(batch,), low=0, high=num_train_images)
            x, xl = train.image[sel], train.label[sel]
            xl = one_hot(xl, nclass)
            v = train_op(x, xl)
            summary.scalar('losses/xe', float(v[0]))
            steps += 1

        # Eval
        accuracy = 0
        for it in trange(0, test.image.shape[0], batch, leave=False, desc='Evaluating'):
            x = test.image[it: it + batch]
            xl = test.label[it: it + batch]
            accuracy += (np.argmax(predict(x), axis=1) == xl).sum()
        accuracy /= test.image.shape[0]
        summary.scalar('eval/accuracy', 100 * accuracy)

        # Use apply_dp_sgd_analysis to compute the current DP guarantee.
        eps = objax.privacy.apply_dp_sgd_analysis(q=sampling_rate,
                                                  noise_multiplier=noise_multiplier,
                                                  steps=steps, delta=delta)
        summary.scalar('privacy/epsilon', eps)
        print('Epoch %04d  Accuracy %.2f Epsilon: %.2f Delta: %.6f ' % (epoch + 1, 100 * accuracy, eps, delta))
        tensorboard.write(summary, step=steps)
