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

"""
MAML implementation to demonstrate gradient of gradient.

https://github.com/ericjang/maml-jax/blob/master/maml.ipynb
"""

import jax.numpy as jn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

import objax


def sample_tasks(outer_batch_size, inner_batch_size):
    # Select amplitude and phase for the task
    amplitudes = []
    phases = []
    for _ in range(outer_batch_size):
        amplitudes.append(np.random.uniform(low=0.1, high=.5))
        phases.append(np.random.uniform(low=0., high=np.pi))

    def get_batch():
        xs, ys = [], []
        for amplitude, phase in zip(amplitudes, phases):
            x = np.random.uniform(low=-5., high=5., size=(inner_batch_size, 1))
            y = amplitude * np.sin(x + phase)
            xs.append(x)
            ys.append(y)
        return np.stack(xs), np.stack(ys)

    x1, y1 = get_batch()
    x2, y2 = get_batch()
    return x1, y1, x2, y2


def make_net():
    return objax.nn.Sequential([
        objax.nn.Linear(1, 40), objax.functional.relu,
        objax.nn.Linear(40, 40), objax.functional.relu,
        objax.nn.Linear(40, 1)
    ])


source = jn.linspace(-5, 5, 100).reshape((100, 1))  # (k, 1)
target = jn.sin(source)

print('Standard training.')
net = make_net()
opt = objax.optimizer.Adam(net.vars())


@objax.Function.with_vars(net.vars())
def loss(x, y):
    return ((y - net(x)) ** 2).mean()


gv = objax.GradValues(loss, net.vars())


@objax.Function.with_vars(net.vars() + opt.vars())
def train_op():
    g, v = gv(source, target)
    opt(0.01, g)
    return v


train_op = objax.Jit(train_op)

for i in range(100):
    train_op()

plt.plot(source, net(source), label='prediction')
plt.plot(source, (target - net(source)) ** 2, label='loss')
plt.plot(source, target, label='target')
plt.legend()
plt.show()

print('MAML training')
net = make_net()
opt = objax.optimizer.Adam(net.vars())


@objax.Function.with_vars(net.vars())
def loss(x, y):
    return ((y - net(x)) ** 2).mean()


gv = objax.GradValues(loss, net.vars())


@objax.Function.with_vars(net.vars())
def maml_loss(x1, y1, x2, y2, alpha=0.1):
    net_vars = net.vars()
    original_weights = net_vars.tensors()  # Save original weights
    g_x1y1 = gv(x1, y1)[0]  # Compute gradient at (x1, y1)
    # Apply gradient update using SGD
    net_vars.assign([v - alpha * g for v, g in zip(original_weights, g_x1y1)])
    loss_x2y2 = loss(x2, y2)
    net_vars.assign(original_weights)  # Restore original weights
    return loss_x2y2


vec_maml_loss = objax.Vectorize(maml_loss, batch_axis=(0, 0, 0, 0, None))


@objax.Function.with_vars(vec_maml_loss.vars())
def batch_maml_loss(x1, y1, x2, y2, alpha=0.1):
    return vec_maml_loss(x1, y1, x2, y2, alpha).mean()


maml_gv = objax.GradValues(batch_maml_loss, vec_maml_loss.vars())


@objax.Function.with_vars(vec_maml_loss.vars() + opt.vars())
def train_op(x1, y1, x2, y2):
    g, v = maml_gv(x1, y1, x2, y2)
    opt(0.001, g)
    return v


train_op = objax.Jit(train_op)

for i in trange(20000, leave=False):
    x1, y1, x2, y2 = sample_tasks(4, 20)
    train_op(x1, y1, x2, y2)

x1 = np.random.uniform(low=-5., high=5., size=(10, 1))
y1 = 1. * np.sin(x1 + 0.)

tensors = net.vars().tensors()
for shot in range(1, 3):
    for v, g in zip(net.vars(), gv(x1, y1)[0]):
        if isinstance(v, objax.TrainVar):
            v.assign(v.value - 0.1 * g)
    plt.plot(source, net(source), label='%d-shot predictions' % shot)
net.vars().assign(tensors)

plt.plot(source, net(source), label='pre-update predictions')
plt.plot(source, target, label='target')
plt.legend()
plt.show()
