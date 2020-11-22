import numpy as np
import tensorflow as tf

import objax
from objax.zoo import vgg

mo = vgg.VGG16()
vgg.load_pretrained_weights_from_keras(mo)
print(mo.vars())

mk = tf.keras.applications.VGG16(weights='imagenet')
x = np.random.randn(4, 3, 224, 224)
yk = mk(x.transpose((0, 2, 3, 1)))  # (4, 1000)

for name, param in ((weight.name, weight.numpy()) for layer in mk.layers for weight in layer.weights):
    print(f'{name:40s} {tuple(param.shape)}')

yo = objax.functional.softmax(mo(x, training=False))
print('Max difference:', np.abs(yk - yo).max())
