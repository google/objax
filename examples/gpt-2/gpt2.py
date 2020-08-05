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

# Thie file is mostly a line-for-line translation from the gpt-2 model from OpenAI.
# The original file was licensed under the MIT license Copyright (c) 2019 OpenAI
# with one additional moodification that reads as follows:
#   We don’t claim ownership of the content you create with GPT-2,
#   so it is yours to do with as you please.
#   We only ask that you use GPT-2 responsibly and clearly indicate
#   your content was created using GPT-2.

import json
import os
import sys

import jax.numpy as np
import numpy as onp

import objax
from objax.variable import TrainVar

if os.path.exists("gpt-2/src"):
    sys.path.append("gpt-2/src")
    import encoder
else:
    print("Error: could not find gpt-2/src/encoder.py")
    print("Please clone GPT-2 into this directory.")
    exit(1)

# Default model architecture parameters
class HParams:
    n_vocab = 0
    n_ctx = 1024
    n_embd = 768
    n_head = 12
    n_layer = 12

# Define the GELU function.
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * (x ** 3))))

# And a logit function, the inverse of sigmoid.
def logit(x):
    x = np.clip(x, 1e-5, 1 - 1e-5)
    return np.log(x / (1 - x))


# Normalization layer used in the transformer
class Norm(objax.module.Module):
    def __init__(self, n_state, axis=-1, epsilon=1e-5):
        super().__init__()
        self.g = TrainVar(np.zeros(n_state))
        self.b = TrainVar(np.ones(n_state))
        self.axis = axis
        self.epsilon = epsilon

    def __call__(self, x):
        # mean across the axis
        u = np.mean(x, axis=self.axis, keepdims=True)
        # standard deviation
        s = np.mean((x - u) ** 2, axis=self.axis, keepdims=True)
        # rescaled values
        x = (x - u) * objax.functional.rsqrt(s + self.epsilon)
        x = x * self.g.value + self.b.value
        return x


def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = x.shape
    return np.reshape(x, start + [n, m // n])


def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = x.shape
    return np.reshape(x, start + [a * b])


# Normalization layer used in the transformer
class Conv1D(objax.module.Module):
    def __init__(self, nx, nf):
        super().__init__()
        self.nx = nx
        self.nf = nf
        self.w = TrainVar(np.zeros([1, nx, nf]))
        self.b = TrainVar(np.zeros([nf]))

    def __call__(self, x):  # NCHW
        *start, nx = x.shape
        assert nx == self.nx

        return np.reshape(np.matmul(np.reshape(x, [-1, self.nx]),
                                    np.reshape(self.w.value, [-1, self.nf])) + self.b.value,
                          start + [self.nf])


def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = np.arange(nd)[:, None]
    j = np.arange(ns)
    m = i >= j - ns + nd
    return np.array(m, dtype=dtype)

# Attention layer used in the transformer
class Attn(objax.module.Module):
    def __init__(self, n_state):
        self.n_state = n_state
        self.conv1 = Conv1D(n_state, n_state * 3)
        self.conv2 = Conv1D(n_state, n_state)

    def __call__(self, x, past):
        assert len(x.shape) == 3  # Should be [batch, sequence, features]
        assert self.n_state % hparams.n_head == 0
        if past is not None:
            assert len(past.shape) == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

        def split_heads(x):
            # From [batch, sequence, features] to [batch, heads, sequence, features]
            return np.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

        def merge_heads(x):
            # Reverse of split_heads
            return merge_states(np.transpose(x, [0, 2, 1, 3]))

        def mask_attn_weights(w):
            # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
            _, _, nd, ns = w.shape
            b = attention_mask(nd, ns, dtype=w.dtype)
            b = np.reshape(b, [1, 1, nd, ns])
            w = w * b - np.array(1e10, w.dtype) * (1 - b)
            return w

        def multihead_attn(q, k, v):
            # q, k, v have shape [batch, heads, sequence, features]
            w = np.matmul(q, np.transpose(k, [0, 1, 3, 2]))
            w = w * objax.functional.rsqrt(np.array(v.shape[-1], w.dtype))

            w = mask_attn_weights(w)
            w = objax.functional.softmax(w)
            a = np.matmul(w, v)
            return a

        c = self.conv1(x)
        q, k, v = map(split_heads, np.split(c, 3, axis=2))
        present = np.stack([k, v], axis=1)
        if past is not None:
            pk, pv = past[:, 0], past[:, 1]
            k = np.concatenate([pk, k], axis=-2)
            v = np.concatenate([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = self.conv2(a)
        return a, present

# Fully connected layer used in the transformer
class MLP(objax.module.Module):
    def __init__(self, nx, n_state):
        self.conv1 = Conv1D(nx, n_state)
        self.conv2 = Conv1D(n_state, nx)

    def __call__(self, x):
        return self.conv2(gelu(self.conv1(x)))


# One (residual) block of transformer
class Block(objax.module.Module):
    def __init__(self, nx, hparams):
        self.nx = nx
        self.norm1 = Norm(hparams.n_embd)
        self.attn = Attn(hparams.n_embd)
        self.norm2 = Norm(hparams.n_embd)
        self.mlp = MLP(nx, nx * 4)

    def __call__(self, x, past):
        assert x.shape[-1] == self.nx

        a, present = self.attn(self.norm1(x), past=past)
        x = x + a
        m = self.mlp(self.norm2(x))
        x = x + m

        return x, present


def past_shape(*, hparams, batch_size=None, sequence=None):
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]


def expand_tile(value, size):
    """Add a new axis of given size."""
    ndims = len(value.shape)
    return np.tile(np.expand_dims(value, axis=0), [size] + [1] * ndims)


def positions_for(tokens, past_length):
    batch_size = tokens.shape[0]
    nsteps = tokens.shape[1]
    return expand_tile(past_length + np.arange(nsteps), batch_size)


# Finally the full tranformer model definition
class Model(objax.module.Module):
    def __init__(self, hparams):
        self.hparams = hparams

        self._wpe = TrainVar(np.zeros([hparams.n_ctx, hparams.n_embd]))
        self._wte = TrainVar(np.zeros([hparams.n_vocab, hparams.n_embd]))

        self.blocks = objax.module.ModuleList([Block(hparams.n_embd, hparams) for _ in range(hparams.n_layer)])
        self.norm = Norm(hparams.n_embd)

    def __call__(self, X, past=None):
        results = {}
        batch, sequence = X.shape

        past_length = 0 if past is None else past.shape[-2]
        h = np.take(self._wte.value, X, axis=0, mode='wrap') + np.take(self._wpe.value, positions_for(X, past_length),
                                                                       axis=0, mode='wrap')

        # Transformer
        presents = []
        if past is not None:
            print(past.shape)
        pasts = np.transpose(past, [1, 0, 2, 3, 4, 5]) if past is not None else [None] * self.hparams.n_layer
        assert len(pasts) == hparams.n_layer
        for block, past in zip(self.blocks, pasts):
            h, present = block(h, past=past)
            presents.append(present)
        results['present'] = np.stack(presents, axis=1)
        h = self.norm(h)

        # Language model loss.  Do tokens <n predict token n?
        h_flat = np.reshape(h, [batch * sequence, hparams.n_embd])
        logits = np.matmul(h_flat, np.transpose(self._wte.value))
        logits = np.reshape(logits, [batch, sequence, hparams.n_vocab])
        results['logits'] = logits
        return results


if __name__ == "__main__":
    hparams = HParams()

    WHICH = sys.argv[1] if len(sys.argv) > 1 else "124M"

    print("Loading model", WHICH)

    if not os.path.exists("gpt-2/models/" + WHICH + "/model.ckpt.meta"):
        print("Model file does not exist in gpt-2/models/" + WHICH + "/model.ckpt.meta")
        print("Please download it and re-run.")
        exit(1)

    with open("gpt-2/models/" + WHICH + "/hparams.json") as f:
        for k, v in json.load(f).items():
            hparams.__dict__[k] = v

    model = objax.Jit(Model(hparams))
    enc = encoder.get_encoder(WHICH, "gpt-2/models")

    # Load the variables from the converted TensorFlow checkpoint

    if not os.path.exists("/tmp/" + WHICH + ".npy"):
        import tensorflow.compat.v1 as tf

        with tf.Session() as sess:
            meta_path = "gpt-2/models/" + WHICH + "/model.ckpt.meta"
            saver = tf.train.import_meta_graph(meta_path)

            saver.restore(sess, "gpt-2/models/" + WHICH + "/model.ckpt")

            r = []
            for v in [x for x in tf.global_variables() if x.name.startswith("model/")]:
                print(v.name, v.shape)
                r.append(v)
            # TODO CLEAN ME UP
            np.save("/tmp/" + WHICH + ".npy", sess.run(r))

    loaded = onp.load("/tmp/" + WHICH + ".npy", allow_pickle=True)
    for (n, v), a in zip(model.vars().items(), loaded):
        if v.value.shape != a.shape:
            raise ValueError(f'Mismatched shapes between v {v.value.shape} and a {a.shape}')
        if isinstance(v, objax.variable.TrainVar):
            v._value = a
        else:
            v.value = a

    print("Enter a prefix to complete:")
    while True:
        print(">>>", end=" ")
        prefix = input()
        print("Continuation: ", end='')
        sys.stdout.flush()

        seq = enc.encode(prefix)

        for _ in range(100):
            out = model(np.array([seq], dtype=np.int32))['logits']
            out = out[0].argmax(1)
            seq.append(int(out[-1]))
            print(enc.decode([seq[-1]]), end='')
            sys.stdout.flush()
