#!/usr/bin/env python

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

"""Extract and save accuracy to 'stats/accuracy.json'.

The accuracy is extracted from the most recent eventfile.
"""

import json
import os

import numpy as np
import tensorflow as tf
from absl import app, flags

FLAGS = flags.FLAGS
TAG = 'accuracy/test'


def summary_dict(accuracies):
    return {'last%02d' % x: np.median(accuracies[-x:]) for x in [1, 10, 20, 50]}


def main(argv):
    if len(argv) > 2:
        raise app.UsageError('Too many command-line arguments.')
    folder = argv[1]
    matches = sorted(tf.io.gfile.glob(os.path.join(folder, 'tb/events.out.tfevents.*')))
    assert matches, 'No events files found'
    tags = set()
    accuracies = []
    for event_file in matches:
        try:
            for e in tf.compat.v1.train.summary_iterator(event_file):
                for v in e.summary.value:
                    if v.tag == TAG:
                        accuracies.append(v.simple_value)
                        break
                    elif not accuracies:
                        tags.add(v.tag)
        except tf.errors.DataLossError:
            continue

    assert accuracies, 'No "%s" tag found. Found tags = %s' % (TAG, tags)
    target_dir = os.path.join(folder, 'stats')
    target_file = os.path.join(target_dir, 'accuracy.json')
    tf.io.gfile.makedirs(target_dir)

    with tf.io.gfile.GFile(target_file, 'w') as f:
        json.dump(summary_dict(accuracies), f, sort_keys=True, indent=4)
    print('Saved: %s' % target_file)


if __name__ == '__main__':
    app.run(main)
