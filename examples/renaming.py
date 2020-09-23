#!/usr/bin/env python3
import re

import jax.numpy as jn

import objax
from objax.util import Renamer


def make_var():
    return objax.TrainVar(jn.zeros([]))


vc = objax.VarCollection({
    '(EfficientNet).stem(ConvBnAct).conv(Conv2d).w': make_var(),
    '(EfficientNet).blocks(Sequential)[0](Sequential)[0](DepthwiseSeparable).conv_dw(Conv2d).w': make_var(),
    '(EfficientNet).blocks(Sequential)[0](Sequential)[0](DepthwiseSeparable).bn_dw(BatchNorm2D).running_mean': make_var(),
    '(EfficientNet).blocks(Sequential)[0](Sequential)[0](DepthwiseSeparable).se(SqueezeExcite).fc1(Conv2d).b': make_var(),
    '(EfficientNet).blocks(Sequential)[0](Sequential)[0](DepthwiseSeparable).se(SqueezeExcite).fc1(Conv2d).w': make_var(),
    '(EfficientNet).head(Head).conv_1x1(Conv2d).w': make_var(),
    '(EfficientNet).head(Head).bn(BatchNorm2D).running_mean': make_var(),
    '(EfficientNet).head(Head).bn(BatchNorm2D).running_var': make_var(),
    '(EfficientNet).head(Head).bn(BatchNorm2D).beta': make_var(),
    '(EfficientNet).head(Head).bn(BatchNorm2D).gamma': make_var(),
    '(EfficientNet).head(Head).classifier(Linear).b': make_var(),
    '(EfficientNet).head(Head).classifier(Linear).w': make_var()
})
print(vc)

print(f'{" Regex renaming ":-^80}')
no_module = re.compile(r'\([^)]+\)')
print(vc.rename(Renamer((no_module, ''))))

print(f'{" Dict renaming ":-^80}')
print(vc.rename(Renamer({'(EfficientNet).': '', '[': '.', ']': ''})))

print(f'{" Function renaming ":-^80}')


def my_rename(x: str) -> str:
    return no_module.sub('', x.replace('(EfficientNet).', '')).replace('[', '.').replace(']', '')


print(vc.rename(Renamer(my_rename)))

print(f'{" Saving/Loading ":-^80}')
print('Saving dict-renamed var collection to disk.')
objax.io.save_var_collection('/tmp/dict_renamed.npz',
                             vc.rename(Renamer({'(EfficientNet).': '', '[': '.', ']': ''})))
try:
    print('Loading dict-renamed var collection to default var collection fails, names mismatch.')
    objax.io.load_var_collection('/tmp/dict_renamed.npz', vc)  # Raise ValueError, missing variables.
    raise Exception('Code should not reach this point')
except ValueError:
    pass
print('Loading dict-renamed var collection to dict-renamed var collection works.')
objax.io.load_var_collection('/tmp/dict_renamed.npz',
                             vc.rename(Renamer({'(EfficientNet).': '', '[': '.', ']': ''})))

print('Loading dict-renamed var collection to function-renamed var collection needs mapping.')
print('  In particular everything remove every module (.*).')
objax.io.load_var_collection('/tmp/dict_renamed.npz',
                             vc.rename(Renamer(my_rename)),
                             Renamer((re.compile(r'\([^)]+\)'), '')))
