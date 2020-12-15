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

__all__ = ['load_var_collection', 'save_var_collection']

import collections
import os
from typing import IO, BinaryIO, Union, Optional

import jax.numpy as jn
import numpy as np
from jax.interpreters.pxla import ShardedDeviceArray

from objax.util import Renamer
from objax.variable import TrainRef, VarCollection


def load_var_collection(file: Union[str, IO[BinaryIO]],
                        vc: VarCollection,
                        renamer: Optional[Renamer] = None):
    """Loads values of all variables in the given variables collection from file.

    Values loaded from file will replace old values in the variables collection.
    If variable exists in the file, but does not exist in the variables collection it will be ignored.
    If variable exists in the variables collection, but not found in the file then exception will be raised.

    Args:
        file: filename or python file handle of the input file.
        vc: variables collection which will be loaded from file.
        renamer: optional renamer to pre-process variables names from the file being read.

    Raises:
        ValueError: if variable from variables collection is not found in the input file.
    """
    renamer = renamer or (lambda x: x)
    do_close = isinstance(file, str)
    if do_close:
        file = open(file, 'rb')
    data = np.load(file, allow_pickle=False)
    name_index = {renamer(k): str(i) for i, k in enumerate(data['names'])}
    name_vars = collections.defaultdict(list)
    for k, v in vc.items():
        if isinstance(v, TrainRef):
            v = v.ref
        name_vars[v].append(k)
    misses = []
    used_vars = set()
    for v, names in name_vars.items():
        for name in names:
            index = name_index.get(name)
            if index is not None:
                used_vars.add(name)
                try:
                    v.assign(jn.array(data[index]))
                except AssertionError as e:
                    raise AssertionError(f'Error when restoring variable {name}: {str(e)}') from None
                break
        else:
            misses += names
    if misses:
        not_used = set(name_index.keys()) - used_vars
        raise ValueError(f'Missing value for variables currently in the model: {misses}. '
                         f'The following variables on disk were not used, '
                         f'maybe the missing variable was renamed from one of these: {not_used}.')
    if do_close:
        file.close()


def save_var_collection(file: Union[str, IO[BinaryIO]], vc: VarCollection):
    """Saves variables collection into file.

    Args:
        file: filename or python file handle of the file where variables will be saved.
        vc: variables collection which will be saved into file.
    """
    do_close = isinstance(file, str)
    if do_close:
        filename, file = file, open(file + '.tmp', 'wb')  # Save to a temporary in case the job is killed while saving.
    data, names, seen, replicated = {}, [], set(), []
    for k, v in vc.items():
        if isinstance(v, TrainRef):
            v = v.ref
        if v not in seen:
            names.append(k)
            data[str(len(data))] = v.value
            seen.add(v)
        if isinstance(v.value, ShardedDeviceArray):
            replicated.append(k)
    if replicated:
        print('Warning: When saving VarCollection, some variables were replicated on multiple devices.')
        print('         While it is valid, in most use cases it is more disk efficient to save variables outside of ')
        print('         vars().replicate().')
    np.savez(file, names=np.array(names), **data)
    if do_close:
        file.close()
        os.rename(filename + '.tmp', filename)  # Atomic rename to avoid broken file (when killed while saving).
