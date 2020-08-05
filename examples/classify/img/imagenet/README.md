# Example of training and evaluation on Imagenet dataset

This example trains a ResNet50 model on the ImageNet2012 dataset.

## Getting data

You have to obtain Imagenet dataset to train the model.

Internally this code uses [TFDS](https://github.com/tensorflow/datasets) which will show download instruction on the first run.
Run `python examples/imagenet/imagenet_train.py` and you will see download instructions, similar to following:

```
AssertionError: Manual directory /home/${USER}/tensorflow_datasets/downloads/manual does not exist or is empty. Create it and download/extract dataset artifacts in there. Additional instructions: manual_dir should contain two files: ILSVRC2012_img_train.tar and
ILSVRC2012_img_val.tar.
```

You have to download data from http://www.image-net.org/download-images and then put it into
the directory mentioned in the message.
On the next run run `imagenet_train.py` will process the data and rearrange it inside data directory which might take a while.
Subsequent runs will re-use already downloaded data.

You can override TFDS data directory by providing `--tfds_data_dir` flag. This might be useful if you don't have enough disk space in the default location or already have a copy of Imagenet data somewhere else.

## Running training

Use following command to run training:

```
python examples/classify/img/imagenet/imagenet_train.py \
    --model_dir="${HOME}/experiments/resnet50"
```

Some additional useful flags include following:

* `--train_device_batch_size` controls per-device training batch size. You may need to adjust it if you don't have enough GPU memory.
* `--eval_device_batch_size` controls per-device evaluation batch size. You may need to adjust it if you don't have enough GPU memory.
* `--eval_every_n_steps` controls number of training steps between evaluation and checkpointing.
* `--tfds_data_dir` overrides directory where TFDS is looking for datasets.
