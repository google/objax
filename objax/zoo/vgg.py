__all__ = ['VGG', 'load_pretrained_weights_from_pytorch', 'vgg11', 'vgg13', 'vgg16', 'vgg19']

from typing import Union, Sequence

import objax
from objax.util.convert import import_weights, pytorch


class VGG(objax.Module):
    def __init__(self, nin: int, nout: int, ops: Sequence[Union[str, int]], use_bn: bool, name: str):
        self.name = name + ('_bn' if use_bn else '')
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
        use_bn = self.name.endswith('_bn')
        name = self.name[:-3] if use_bn else self.name
        return f'{self.__class__.__name__}(nin={self.features[0].w.value.shape[2]}, ' \
               f'nout={self.features[0].w.value.shape[3]}, ops={self.ops}, use_bn={use_bn}, name={repr(name)})'


def load_pretrained_weights_from_pytorch(m: VGG):
    import torchvision
    torch_model = getattr(torchvision.models, m.name)(pretrained=True)
    torch_model.eval()  # Just a safety precaution.
    numpy_arrays = {name: param.numpy() for name, param in torch_model.state_dict().items()}
    numpy_names = {k: pytorch.rename(k) for k in m.vars().keys()}
    import_weights(m.vars(), numpy_arrays, numpy_names, pytorch.ARRAY_CONVERT)


def vgg11(use_bn: bool):
    ops = 64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'
    return VGG(3, 1000, ops, use_bn=use_bn, name='vgg11')


def vgg13(use_bn: bool):
    ops = 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'
    return VGG(3, 1000, ops, use_bn=use_bn, name='vgg13')


def vgg16(use_bn: bool):
    ops = 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'
    return VGG(3, 1000, ops, use_bn=use_bn, name='vgg16')


def vgg19(use_bn: bool):
    ops = 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'
    return VGG(3, 1000, ops, use_bn=use_bn, name='vgg19')
