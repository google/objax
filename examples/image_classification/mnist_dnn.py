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

# *Note*: The purpose of the example on MNIST is to demonstrate the use of a deep
# neural network for classification. As such, the network does not achieve State
# of the Art (SOTA) classification accurary. A Convolutional Neural Network (CNN)
# should be used for that purpose.

import os

import numpy as np
import tensorflow_datasets as tfds
from tqdm import trange

import objax
from objax.functional import leaky_relu, one_hot
from objax.jaxboard import SummaryWriter, Summary
from objax.util import EasyDict
from objax.zoo.dnnet import DNNet

# Data
DATA_DIR = os.path.join(os.environ['HOME'], 'TFDS')
data = tfds.as_numpy(tfds.load(name='mnist', batch_size=-1, data_dir=DATA_DIR))
train_size = len(data['train']['image'])
test_size = len(data['test']['image'])
train_shape = data['train']['image'].shape
image_size = train_shape[1] * train_shape[2] * train_shape[3]
nclass = len(np.unique(data['train']['label']))
flat_train_images = np.reshape(data['train']['image'].transpose(0, 3, 1, 2) / 127.5 - 1,
                               (train_size, image_size))
flat_test_images = np.reshape(data['test']['image'].transpose(0, 3, 1, 2) / 127.5 - 1, (test_size, image_size))
test = EasyDict(image=flat_test_images, label=data['test']['label'])
train = EasyDict(image=flat_train_images, label=data['train']['label'])
del data

# Settings
lr = 0.0002
batch = 64
num_train_epochs = 40
dnn_layer_sizes = image_size, 128, 10
logdir = f'experiments/classify/img/mnist/filters{dnn_layer_sizes}'

# Model
model = DNNet(dnn_layer_sizes, leaky_relu)
model_vars = model.vars()
opt = objax.optimizer.Adam(model_vars)
ema = objax.optimizer.ExponentialMovingAverage(model_vars, momentum=0.999)
predict = objax.Jit(ema.replace_vars(lambda x: objax.functional.softmax(model(x, training=False))),
                    model_vars + ema.vars())


def loss(x, label):
    logit = model(x)
    return objax.functional.loss.cross_entropy_logits(logit, label).mean()


gv = objax.GradValues(loss, model.vars())


def train_op(x, xl):
    g, v = gv(x, xl)  # returns gradients, loss
    opt(lr, g)
    ema()
    return v


# gv.vars() contains the model variables.
train_op = objax.Jit(train_op, gv.vars() + opt.vars() + ema.vars())

# Training
print(model_vars)
print(f'Visualize results with: tensorboard --logdir "{logdir}"')
print("Disclaimer: This code demonstrates the DNNet class. For SOTA accuracy use a CNN instead.")
with SummaryWriter(os.path.join(logdir, 'tb')) as tensorboard:
    for epoch in range(num_train_epochs):
        # Train one epoch
        summary = Summary()
        loop = trange(0, train_size, batch,
                      leave=False, unit='img', unit_scale=batch,
                      desc='Epoch %d/%d' % (1 + epoch, num_train_epochs))
        for it in loop:
            sel = np.random.randint(size=(batch,), low=0, high=train_size)
            x, xl = train.image[sel], train.label[sel]
            xl = one_hot(xl, nclass)
            v = train_op(x, xl)
            summary.scalar('losses/xe', float(v[0]))

        # Eval
        accuracy = 0
        for it in trange(0, test.image.shape[0], batch, leave=False, desc='Evaluating'):
            x = test.image[it: it + batch]
            xl = test.label[it: it + batch]
            accuracy += (np.argmax(predict(x), axis=1) == xl).sum()
        accuracy /= test.image.shape[0]
        summary.scalar('eval/accuracy', 100 * accuracy)
        print('Epoch %04d  Loss %.2f  Accuracy %.2f' % (epoch + 1, summary['losses/xe'](), summary['eval/accuracy']()))

        tensorboard.write(summary, step=(epoch + 1) * train_size)
