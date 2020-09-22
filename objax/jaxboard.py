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

import enum
import os
from time import time
from typing import Union, Callable, Tuple, ByteString

import numpy as np
from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto import summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.util.tensor_util import make_tensor_proto

from objax import util


class Reducer(enum.Enum):
    """Reduces tensor batch into a single tensor."""
    FIRST = lambda x: x[0]
    LAST = lambda x: x[-1]
    MEAN = lambda x: np.mean(x)


class DelayedScalar:
    def __init__(self, reduce: Union[Callable, Reducer]):
        self.values = []
        self.reduce = reduce

    def __call__(self):
        return self.reduce(self.values)


class Image:
    def __init__(self, shape: Tuple[int, int, int], png: ByteString):
        self.shape = shape
        self.png = png


class Text:
    def __init__(self, text: str):
        self.text = text


class Summary(dict):
    """Writes entries to `Summary` protocol buffer."""

    def image(self, tag: str, image: np.ndarray):
        """Adds image to the summary. Float image in [-1, 1] in CHW format expected."""
        self[tag] = Image(image.shape, util.image.to_png(image))

    def scalar(self, tag: str, value: float, reduce: Union[Callable, Reducer] = Reducer.MEAN):
        """Adds scalar to the summary."""
        if tag not in self:
            self[tag] = DelayedScalar(reduce)
        self[tag].values.append(value)

    def text(self, tag: str, text: str):
        """Adds text to the summary."""
        self[tag] = Text(text)

    def __call__(self):
        entries = []
        for tag, value in self.items():
            if isinstance(value, DelayedScalar):
                entries.append(summary_pb2.Summary.Value(tag=tag, simple_value=value()))
            elif isinstance(value, Image):
                image_summary = summary_pb2.Summary.Image(encoded_image_string=value.png,
                                                          colorspace=value.shape[0],
                                                          height=value.shape[1],
                                                          width=value.shape[2])
                entries.append(summary_pb2.Summary.Value(tag=tag, image=image_summary))
            elif isinstance(value, Text):
                metadata = summary_pb2.SummaryMetadata(
                    plugin_data=summary_pb2.SummaryMetadata.PluginData(plugin_name='text'))
                entries.append(summary_pb2.Summary.Value(tag=tag, metadata=metadata,
                                                         tensor=make_tensor_proto(values=value.text.encode('utf-8'),
                                                                                  shape=(1,))))
            else:
                raise NotImplementedError(tag, value)
        return summary_pb2.Summary(value=entries)


class SummaryWriter:
    """Writes entries to event files in the logdir to be consumed by Tensorboard."""

    def __init__(self, logdir: str, queue_size: int = 5, write_interval: int = 5):
        """Creates SummaryWriter instance.

        Args:
            logdir: directory where event file will be written.
            queue_size: size of the queue for pending events and summaries
                        before one of the 'add' calls forces a flush to disk.
            write_interval: how often, in seconds, to write the pending events and summaries to disk.
        """
        if not os.path.isdir(logdir):
            os.makedirs(logdir, exist_ok=True)

        self.writer = EventFileWriter(logdir, queue_size, write_interval)

    def write(self, summary: Summary, step: int):
        """Adds on event to the event file."""
        self.writer.add_event(event_pb2.Event(step=step, summary=summary(), wall_time=time()))

    def close(self):
        """Flushes the event file to disk and close the file."""
        self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
