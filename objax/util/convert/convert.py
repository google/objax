__all__ = ['assign', 'import_weights']

import re
from typing import Dict, Callable

import jax.numpy as jn
import numpy as np

from objax.variable import BaseVar, VarCollection


def assign(x: BaseVar, v: np.ndarray):
    x.assign(jn.array(v.reshape(x.value.shape)))


def import_weights(target_vc: VarCollection,
                   source_numpy: Dict[str, np.ndarray],
                   target_to_source_names: Dict[str, str],
                   numpy_convert: Dict[str, Callable[[BaseVar, np.ndarray], None]]):
    module_var = re.compile(r'.*(\([^)]*\)\.[^(]*)$')
    for k, v in target_vc.items():
        s = target_to_source_names[k]
        t = module_var.match(k).group(1)
        if s not in source_numpy:
            print(f'Skipping {k} ({s})')
            continue
        assert t in numpy_convert, f'Unhandled name {k}'
        numpy_convert[t](v, source_numpy[s])
