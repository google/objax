# Example of training and evaluation of ResNet50 on Imagenet

This example trains a ResNet50 model on the ImageNet2012 dataset.

## Getting data

You have to obtain the Imagenet dataset to train the model.

Internally this code uses [TFDS](https://github.com/tensorflow/datasets) which will show download instructions on the first run.
Run `python examples/image_classification/imagenet_resnet50_train.py` and you will see download instructions, similar to the following:

```
AssertionError: Manual directory /home/${USER}/tensorflow_datasets/downloads/manual does not exist or is empty. Create it and download/extract dataset artifacts in there. Additional instructions: manual_dir should contain two files: ILSVRC2012_img_train.tar and
ILSVRC2012_img_val.tar.
```

You have to download data from http://www.image-net.org/download-images and then put it into
the directory mentioned in the message.
On the next run, run `imagenet_resnet50_train.py` which will process the data and rearrange it inside the data directory which might take a while.
Subsequent runs will re-use the already downloaded data.

You can override TFDS data directory by providing the `--tfds_data_dir` flag. This might be useful if you don't have enough disk space in the default location or already have a copy of Imagenet data somewhere else.

## Training the model

Use the following command to train:

```
python examples/classify/img/imagenet/imagenet_train.py \
    --model_dir="${HOME}/experiments/resnet50"
```

Some additional useful flags include the following:

* `--train_device_batch_size` controls per-device training batch size. You may need to adjust it if you don't have enough GPU memory.
* `--eval_device_batch_size` controls per-device evaluation batch size. You may need to adjust it if you don't have enough GPU memory.
* `--eval_every_n_steps` controls the number of training steps between evaluation and checkpointing.
* `--tfds_data_dir` overrides the directory where TFDS looks for datasets.
