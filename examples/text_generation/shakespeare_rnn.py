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

import collections
import random
import re

import jax
import jax.numpy as jn
import tensorflow_datasets as tfds

import objax
from objax.functional import one_hot
<<<<<<< HEAD:examples/text_generation/shakespeare_rnn.py
from objax.nn import SimpleRNN
=======
from objax.nn import RNN
>>>>>>> 2c04d4e (Move RNN to layers.py and make it stateless.):examples/rnn/shakespeare.py


def tokenize(lines, token_type='word'):
    """Split the lines list into word or char tokens depending on token_type."""

    if token_type == 'word':
        return [line.split(' ') for line in lines]
    elif token_type == 'char':
        return [list(line) for line in lines]
    else:
        raise ValueError('ERROR: unknown token type', token_type)


def count_corpus(token_list):
    """Return a Counter of the tokens in token_list.

    Args:
        token_list: list of token lists
    """
    tokens = [tk for tokens in token_list for tk in tokens]
    return collections.Counter(tokens)


class Vocabulary:
    """Vocabulary extracts set of unique tokens and
    constructs token to index and index to token lookup tables.
    """

    def __init__(self, token_list):
        counter = count_corpus(token_list)

        self.token_freqs = sorted(counter.items(), key=lambda x: x[0])
        self.token_freqs.sort(key=lambda x: x[1], reverse=True)
        self.unk, uniq_tokens = 0, ['<unk>']
        uniq_tokens += [token for token, freq in self.token_freqs]
        self.idx_to_token, self.token_to_idx = [], dict()

        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]


def seq_data_iter(corpus, batch_size, num_steps):
    # Offset the iterator over the data for uniform starts
    corpus = corpus[random.randint(0, num_steps):]
    # Subtract 1 extra since we need to account for label
    num_examples = ((len(corpus) - 1) // num_steps)
    example_indices = list(range(0, num_examples * num_steps, num_steps))
    random.shuffle(example_indices)

    def data(pos):
        # This returns a sequence of length `num_steps` starting from `pos`
        return corpus[pos: pos + num_steps]

    # Discard half empty batches
    num_batches = num_examples // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # `batch_size` indicates the random examples read each time
        batch_indices = example_indices[i:(i + batch_size)]
        X = [data(j) for j in batch_indices]
        Y = [data(j + 1) for j in batch_indices]
        yield X, Y


class DataLoader:
    """An iterator to load sequence data."""

    def __init__(self, batch_size, num_steps, token_type):
        self.data_iter_fn = seq_data_iter
        self.corpus, self.vocab = load_shakespeare_corpus(token_type)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_shakespeare_corpus(token_type='char'):
    """Load the tiny_shakespeare TFDS and return its corpus and vodabulary."""
    data = tfds.as_numpy(tfds.load(name='tiny_shakespeare', batch_size=-1))
    train_data = data['train']
    input_string = train_data['text'][0].decode()  # decode binary string
    re.sub('[^A-Za-z]+', ' ', input_string.strip().lower())
    lines = input_string.splitlines()

    token_list = tokenize(lines, token_type)

    vocab = Vocabulary(token_list)
    corpus = [vocab[tk] for tokens in token_list for tk in tokens]
    return corpus, vocab


def load_shakespeare(batch_size, num_steps, token_type):
    data_iter = DataLoader(batch_size, num_steps, token_type)
    return data_iter, data_iter.vocab


batch_size, num_steps = 10, 10
num_epochs = 500
num_hiddens = 256
lr = 0.0001
theta = 1

train_iter, vocab = load_shakespeare(batch_size, num_steps, 'char')
vocab_size = len(vocab)

model = SimpleRNN(num_hiddens, vocab_size, vocab_size)
model_vars = model.vars()

# Sample call for forward pass
X = jn.arange(batch_size * num_steps).reshape(batch_size, num_steps).T
X_one_hot = one_hot(X, vocab_size)
Z, _ = model(X_one_hot)


def predict_char(prefix, num_predicts, model, vocab):
    outputs = [vocab[prefix[0]]]
    get_input = lambda: one_hot(jn.array([outputs[-1]]).reshape(1, 1), len(vocab))
    for y in prefix[1:]:  # Warmup state with prefix
        model(get_input())
        outputs.append(vocab[y])
    for _ in range(num_predicts):  # Predict num_predicts steps
        Y, _ = model(get_input())
        outc = int(Y.argmax(axis=1).reshape(1))
        outputs.append(outc)
    return ''.join([vocab.idx_to_token[i] for i in outputs])


print(predict_char('to be or not to be', 10, model, vocab))

opt = objax.optimizer.Adam(model_vars)
ema = objax.optimizer.ExponentialMovingAverage(model_vars, momentum=0.999)


def loss(x, label):  # sum(label * log(softmax(logit)))
    logits, _ = model(x)
    return objax.functional.loss.cross_entropy_logits(logits, label).mean()


gv = objax.GradValues(loss, model.vars())


def clip_gradients(grads, theta):
    total_grad_norm = jn.linalg.norm([jn.linalg.norm(g) for g in grads])
    scale_factor = jn.minimum(theta / total_grad_norm, 1.)
    return [g * scale_factor for g in grads]


def train_op(x, xl):
    g, v = gv(x, xl)  # returns gradients, loss
    clipped_g = clip_gradients(g, theta)
    opt(lr, clipped_g)
    ema()
    return v


train_op = objax.Jit(train_op, gv.vars() + opt.vars() + ema.vars())

# Training
for epoch in range(num_epochs):
    for test_data, labels in train_iter:
        X = jn.array(test_data).T
        X_one_hot = one_hot(X, vocab_size)
        Y = jn.array(labels).T
        Y_one_hot = one_hot(Y, vocab_size)
        flat_labels = jn.concatenate(Y_one_hot, axis=0)
        v = train_op(X_one_hot, flat_labels)
    if epoch % 10 == 0:
        print("loss:", float(v[0]))

print(predict_char('to be or not to be', 40, model, vocab))
