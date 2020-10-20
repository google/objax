[home](../../README.md) > [examples](../README.md) > image_classification

# Image Classification Examples

This directory contains various classification examples on image datasets:

* [MNIST](http://yann.lecun.com/exdb/mnist/):

  * `mnist_dnn.py` - simple MNIST classification example.
    *Note*: The purpose of the example on MNIST is to demonstrate the use of a deep
    neural network for classification. As such, the network does not achieve State
    of the Art (SOTA) classification accurary. A Convolutional Neural Network (CNN)
    should be used for that purpose.

  * `mnist_cnn.py` - a CNN-based MNIST classification example.

  * `mnist_dp.py` - MNIST example with differential privacy.

* [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)

  * `cifar10_simple.py` - very simple CIFAR10 classification example which
    demonstrated how to write basic training loop with data augmentation

  * `cifar10_advanced.py` - more advanced CIFAR10 example which allows user to configure
    neural network architecture and other hyperparameters. It also supports training on multiple
    GPUs using `objax.Parallel`.

* [Imagenet](http://www.image-net.org/challenges/LSVRC/2012/)

  * `imagenet_pretrained_vgg.py` - example which shows how to load pre-trained weights for a VGG model and use it
    to classify input images. For more details see [documentation](pretrained_vgg.md).

  * `imagenet_resnet50_train.py` - example which shows how to train Resnet50 model on Imagenet.
    For more details see example [documentation](imagenet/README.md).

* [Horses or Humans](https://www.kaggle.com/sanikamal/horses-or-humans-dataset)

  * `horses_logistic.py` - simple example using logistic regression.