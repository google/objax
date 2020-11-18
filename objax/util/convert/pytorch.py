__all__ = ['ARRAY_CONVERT', 'rename']

import re

from objax.util.convert import assign

ARRAY_CONVERT = {
    '(BatchNorm2D).beta': assign,
    '(BatchNorm2D).gamma': assign,
    '(BatchNorm2D).running_mean': assign,
    '(BatchNorm2D).running_var': assign,
    '(Conv2D).b': assign,
    '(Conv2D).w': lambda x, y: assign(x, y.transpose((2, 3, 1, 0))),
    '(Linear).b': assign,
    '(Linear).w': lambda x, y: assign(x, y.T),
}


def rename(x):
    x = x.replace('(BatchNorm2D).gamma', '(BatchNorm2D).weight').replace('(BatchNorm2D).beta', '(BatchNorm2D).bias')
    x = re.sub(r'\([^)]*\)', '', x)
    x = re.sub(r'^\.', '', x)
    x = re.sub('.w$', '.weight', x)
    x = re.sub('.b$', '.bias', x)
    x = x.replace('[', '.').replace(']', '')
    return x
