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

__all__ = ['Checkpoint']

import glob
import os
from typing import Callable, Optional

from objax.io.ops import load_var_collection, save_var_collection
from objax.typing import FileOrStr
from objax.variable import VarCollection


class Checkpoint:
    """Helper class which performs saving and restoring of the variables.
    
    Variables are stored in the checkpoint files. One checkpoint file stores a single snapshot of the variables.
    Different checkpoint files store different snapshots of the variables (for example at different training step).
    Each checkpoint has associated index, which is used to identify time when snapshot of the variables was made.
    Typically training step or training epoch are used as an index.
    """

    DIR_NAME: str = 'ckpt'
    """Name of the subdirectory of model directory where checkpoints will be saved."""

    FILE_MATCH: str = '*.npz'
    """File pattern which is used to search for checkpoint files."""

    FILE_FORMAT: str = '%010d.npz'
    """Format of the filename of one checkpoint file."""

    LOAD_FN: Callable[[FileOrStr, VarCollection], None] = staticmethod(load_var_collection)
    """Load function, which loads variables collection from given file."""

    SAVE_FN: Callable[[FileOrStr, VarCollection], None] = staticmethod(save_var_collection)
    """Save function, which saves variables collection into given file."""

    def __init__(self, logdir: str, keep_ckpts: int, makedir: bool = True, verbose: bool = True):
        """Creates instance of the Checkpoint class.

        Args:
            logdir: model directory. Checkpoints will be saved in the subdirectory of model directory.
            keep_ckpts: maximum number of checkpoints to keep.
            makedir: if True then directory for checkpoints will be created,
                otherwise it's expected that directory already exists.
            verbose: if True then print when data is restored from checkpoint.
        """
        self.logdir = logdir
        self.keep_ckpts = keep_ckpts
        self.verbose = verbose
        if makedir:
            os.makedirs(os.path.join(logdir, self.DIR_NAME), exist_ok=True)

    @staticmethod
    def checkpoint_idx(filename: str):
        """Returns index of checkpoint from given checkpoint filename.

        Args:
            filename: checkpoint filename.

        Returns:
            checkpoint index.
        """
        return int(os.path.basename(filename).split('.')[0])

    def restore(self, vc: VarCollection, idx: Optional[int] = None):
        """Restores values of all variables of given variables collection from the checkpoint.

        Old values from the variables collection will be replaced with the new values read from checkpoint.
        If variable does not exist in the variables collection, it won't be restored from checkpoint.

        Args:
            vc: variables collection to restore.
            idx: if provided then checkpoint index to use, if None then latest checkpoint will be restored.

        Returns:
            idx: index of the restored checkpoint.
            ckpt: full path to the restored checkpoint.
        """
        if idx is None:
            all_ckpts = glob.glob(os.path.join(self.logdir, self.DIR_NAME, self.FILE_MATCH))
            if not all_ckpts:
                return 0, ''
            idx = self.checkpoint_idx(max(all_ckpts))
        ckpt = os.path.join(self.logdir, self.DIR_NAME, self.FILE_FORMAT % idx)
        if self.verbose:
            print('Resuming from', ckpt)
        self.LOAD_FN(ckpt, vc)
        return idx, ckpt

    def save(self, vc: VarCollection, idx: int):
        """Saves variables collection to checkpoint with given index.
        
        Args:
            vc: variables collection to save.
            idx: index of the new checkpoint where variables should be saved.
        """
        self.SAVE_FN(os.path.join(self.logdir, self.DIR_NAME, self.FILE_FORMAT % idx), vc)
        for ckpt in sorted(glob.glob(os.path.join(self.logdir, self.DIR_NAME, self.FILE_MATCH)))[:-self.keep_ckpts]:
            os.remove(ckpt)
