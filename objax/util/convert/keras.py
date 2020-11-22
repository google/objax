__all__ = ['ARRAY_CONVERT', 'rename']

import re

from .convert import assign

ARRAY_CONVERT = {
    # '(BatchNorm2D).beta': assign,
    # '(BatchNorm2D).gamma': assign,
    # '(BatchNorm2D).running_mean': assign,
    # '(BatchNorm2D).running_var': assign,
    '(Conv2D).b': assign,
    '(Conv2D).w': assign,
    '(Linear).b': assign,
    '(Linear).w': assign,
}


def rename(x):
    # x = x.replace('(BatchNorm2D).gamma', '(BatchNorm2D).weight').replace('(BatchNorm2D).beta', '(BatchNorm2D).bias')
    x = re.sub(r'\([^)]*\)', '', x)
    x = re.sub(r'^\.', '', x)
    x = re.sub(r'.w$', '/kernel', x)
    x = re.sub(r'.b$', '/bias', x)
    x = re.sub(r'\[|\]', '', x)
    x = re.sub(r'\.', '_', x)
    return x
