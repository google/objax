# Image Classification with Pretrained VGG model

This [example](pretrained_vgg.py) demonstrates how to run image classification with
[VGG-19](https://www.robots.ox.ac.uk/~vgg/publications/2015/Simonyan15/simonyan15.pdf) model using
weights pretrained on [ImageNet dataset](http://www.image-net.org/).

## Getting weights of VGG-19 pretrained model

Please download the weights of VGG-19 pretrained model from this
[link](https://mega.nz/file/xZ8glS6J#MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs) and copy to
`./objax/zoo/pretrained/vgg19.npy`.

## Classifying images

This [example](pretrained_vgg.py) shows how to classifying an image downloaded from the internet.
You can set an `IMAGE_PATH` to classify your own image.
