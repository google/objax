__all__ = ['VGG', 'VGG11', 'VGG11_BN', 'VGG13', 'VGG13_BN', 'VGG16', 'VGG16_BN', 'VGG19', 'VGG19_BN',
           'load_pretrained_weights_from_keras', 'load_pretrained_weights_from_pytorch']

from typing import Union, Sequence

import objax
from objax.util.convert import import_weights, pytorch

OPS = dict(
    vgg11=(64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'),
    vgg13=(64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'),
    vgg16=(64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'),
    vgg19=(64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M')
)


class VGG(objax.Module):
    def __init__(self, nin: int, nout: int, ops: Sequence[Union[str, int]], use_bn: bool):
        self.use_bn = use_bn
        self.ops = tuple(ops)
        n = nin
        self.features = objax.nn.Sequential()
        for v in ops:
            if v == 'M':
                self.features.append(objax.functional.max_pool_2d)
                continue
            self.features.append(objax.nn.Conv2D(n, v, 3, padding=1))
            if use_bn:
                self.features.append(objax.nn.BatchNorm2D(v, momentum=0.1, eps=1e-5))
            self.features.append(objax.functional.relu)
            n = v

        self.classifier = objax.nn.Sequential([objax.nn.Linear(512 * 7 * 7, 4096), objax.functional.relu,
                                               objax.nn.Dropout(0.5),
                                               objax.nn.Linear(4096, 4096), objax.functional.relu,
                                               objax.nn.Dropout(0.5),
                                               objax.nn.Linear(4096, nout)])

    def __call__(self, *args, **kwargs):
        features = objax.functional.flatten(self.features(*args, **kwargs))
        return self.classifier(features, **kwargs)

    def __repr__(self):
        nin, nout = self.features[0].w.value.shape[2:]
        return f'{self.__class__.__name__}(nin={nin}, nout={nout}, ops={self.ops}, use_bn={self.use_bn})'


class CustomVGG(VGG):
    def __repr__(self):
        nin, nout = self.features[0].w.value.shape[2:]
        return f'{self.__class__.__name__}(nin={nin}, nout={nout})'


class VGG11(CustomVGG):
    def __init__(self, nin: int = 3, nout: int = 1000):
        super().__init__(nin, nout, ops=OPS['vgg11'], use_bn=False)


class VGG11_BN(CustomVGG):
    def __init__(self, nin: int = 3, nout: int = 1000):
        super().__init__(nin, nout, ops=OPS['vgg11'], use_bn=True)


class VGG13(CustomVGG):
    def __init__(self, nin: int = 3, nout: int = 1000):
        super().__init__(nin, nout, ops=OPS['vgg13'], use_bn=False)


class VGG13_BN(CustomVGG):
    def __init__(self, nin: int = 3, nout: int = 1000):
        super().__init__(nin, nout, ops=OPS['vgg13'], use_bn=True)


class VGG16(CustomVGG):
    def __init__(self, nin: int = 3, nout: int = 1000):
        super().__init__(nin, nout, ops=OPS['vgg16'], use_bn=False)


class VGG16_BN(CustomVGG):
    def __init__(self, nin: int = 3, nout: int = 1000):
        super().__init__(nin, nout, ops=OPS['vgg16'], use_bn=True)


class VGG19(CustomVGG):
    def __init__(self, nin: int = 3, nout: int = 1000):
        super().__init__(nin, nout, ops=OPS['vgg19'], use_bn=False)


class VGG19_BN(CustomVGG):
    def __init__(self, nin: int = 3, nout: int = 1000):
        super().__init__(nin, nout, ops=OPS['vgg19'], use_bn=True)


def load_pretrained_weights_from_keras(m: VGG):
    import tensorflow as tf
    assert hasattr(tf.keras.applications, m.__class__.__name__), \
        f'No Keras pretrained model for {m.__class__.__name__}'
    # 1. Get Keras model
    keras_model = getattr(tf.keras.applications, m.__class__.__name__)(weights='imagenet')

    # 2. Get Keras model weights
    keras_numpy = {weight.name.split(':')[0]: weight.numpy()  # Remove :0 at the end of the variable name.
                   for layer in keras_model.layers for weight in layer.weights}
    # 2.1 Flattening differs between NHWC and NCHW: convert first linear layer post-flattening.
    nhwc_kernel = keras_numpy['fc1/kernel'].reshape((7, 7, 512, 4096))
    keras_numpy['fc1/kernel'] = nhwc_kernel.transpose((2, 0, 1, 3)).reshape((-1, 4096))

    # 3. Map Objax names to Keras names.
    #    The architectures are syntactically different (Objax uses Sequential while Keras does not).
    #    So we have to map the name semi-manually since there's no automatic way to do it.
    keras_names = {k: objax.util.convert.keras.rename(k) for k in m.vars().keys()}
    to_keras = {
        'classifier0': 'fc1',
        'classifier3': 'fc2',
        'classifier6': 'predictions',
    }

    # The features in Keras are of the form "block{i}_conv{j}/variable"
    # In Objax they are of the form "features{pos}/variable"
    # Below we convert list position to block_conv.
    target_to_source_names = {}
    block_id, conv_id, seq_id = 1, 1, 0
    for k, v in keras_names.items():
        if '/' not in v:
            target_to_source_names[k] = v
            continue
        layer, variable = v.split('/')
        if layer.startswith('features'):
            new_seq_id = int(layer[8:])
            if new_seq_id - seq_id == 2:
                conv_id += 1
            elif new_seq_id - seq_id == 3:
                block_id += 1
                conv_id = 1
            else:
                assert new_seq_id == seq_id
            seq_id = new_seq_id
            target_to_source_names[k] = f'block{block_id}_conv{conv_id}/{variable}'
        elif layer.startswith('classifier'):
            target_to_source_names[k] = f'{to_keras[layer]}/{variable}'
        else:
            target_to_source_names[k] = v

    objax.util.convert.import_weights(m.vars(), keras_numpy, target_to_source_names,
                                      objax.util.convert.keras.ARRAY_CONVERT)


def load_pretrained_weights_from_pytorch(m: VGG):
    import torchvision
    assert hasattr(torchvision.models, m.__class__.__name__.lower()), \
        f'No TorchVision pretrained model for {m.__class__.__name__.lower()}'
    torch_model = getattr(torchvision.models, m.__class__.__name__.lower())(pretrained=True)
    torch_model.eval()  # Just a safety precaution.
    torch_numpy = {name: param.numpy() for name, param in torch_model.state_dict().items()}
    target_to_source_names = {k: pytorch.rename(k) for k in m.vars().keys()}
    import_weights(m.vars(), torch_numpy, target_to_source_names, pytorch.ARRAY_CONVERT)
