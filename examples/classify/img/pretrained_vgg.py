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
from urllib import request

import jax.numpy as jn
import numpy as np
from PIL import Image

import objax
from objax.zoo import vgg

IMAGE_URL = 'https://upload.wikimedia.org/wikipedia/commons/b/b0/Bengal_tiger_%28Panthera_tigris_tigris%29_female_3_crop.jpg'
IMAGE_PATH = './examples/classify/img/misc/001.jpg'
SYNSET_PATH = './objax/zoo/pretrained/synset.txt'

# Load input image.
if not os.path.exists(os.path.dirname(IMAGE_PATH)):
    os.makedirs(os.path.dirname(IMAGE_PATH))
request.urlretrieve(IMAGE_URL, IMAGE_PATH)
img = Image.open(IMAGE_PATH)
img = np.array(img.resize((224, 224))).astype(np.float32)
img = jn.array(img).transpose((2, 0, 1))[None,]

# Load model with pretrained weights and make a prediction.
model = vgg.VGG19(pretrained=True)
logit = model(img)
prob = objax.functional.softmax(logit)[0]

# Present prediction output.
synset = [l.strip() for l in open(SYNSET_PATH).readlines()]
pred = jn.argsort(prob)[::-1][:5]
for i in range(5):
    print('Top {:d} (prob {:.3f}) {}'.format(i + 1, prob[pred[i]], synset[pred[i]]))
