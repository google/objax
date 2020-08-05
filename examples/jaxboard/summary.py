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

import numpy as np

import objax

LOGDIR = 'experiments/summary_test/tb'
with objax.jaxboard.SummaryWriter(LOGDIR) as tensorboard:
    summary = objax.jaxboard.Summary()
    summary.text('text', '<pre>Hello this just text\nand a newline</pre>')
    summary.text('html', '<table><tr><th>col1</th><th>col2</th></tr>'
                         '<tr><td>row1.1</td><td>row1.2</td></tr>'
                         '<tr><td>row2.1</td><td>row2.2</td></tr></table>')
    img = np.zeros((3, 32, 32), 'f')
    img[0] += np.linspace(-1, 1, 32)
    img[1] += np.linspace(-1, 1, 32)[:, None]
    img[2] += np.linspace(-1, 1, 32)[:, None] * np.linspace(-1, 1, 32)
    summary.image('image', img)
    summary.scalar('avg', 0)
    summary.scalar('avg', 1)
    summary.scalar('avg', 2)
    summary.scalar('avg', 3)
    tensorboard.write(summary, step=1)

    summary = objax.jaxboard.Summary()
    summary.text('text', '<pre>Hello this just text\nat step 2</pre>')
    summary.scalar('avg', 4)
    summary.scalar('avg', 7)
    tensorboard.write(summary, step=2)

    print(f'Saved to {LOGDIR}')
    print(f'Visualize with: tensorboard --logdir "{LOGDIR}"')
