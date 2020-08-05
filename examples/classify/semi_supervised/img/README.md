[home](../../../../README.md) > [examples](../../../../README.md) > [classify](../../../README.md) > [semi_supervised](../../README.md) > img

# Semi-Supervised Image Classification 

This directory contains several semi-supervised learning techniques:

* [FixMatch](https://arxiv.org/abs/2001.07685)
* More to come


## Setup

### Required environment variables

```bash
export PYTHONPATH=$PYTHONPATH:.
export ML_DATA="path to where you want the datasets saved"
export PROJECT="ObjaxSSL"
export SSL_PATH=examples/classify/semi_supervised/img
```

## Data preparation

```bash
# Download datasets
CUDA_VISIBLE_DEVICES= $SSL_PATH/scripts/create_datasets.py
cp $ML_DATA/$PROJECT/svhn-test.tfrecord $ML_DATA/$PROJECT/svhnx-test.tfrecord

# Create unlabeled datasets
CUDA_VISIBLE_DEVICES= $SSL_PATH/scripts/create_unlabeled.py $ML_DATA/$PROJECT/SSL/cifar10 $ML_DATA/$PROJECT/cifar10-train.tfrecord &
CUDA_VISIBLE_DEVICES= $SSL_PATH/scripts/create_unlabeled.py $ML_DATA/$PROJECT/SSL/cifar100 $ML_DATA/$PROJECT/cifar100-train.tfrecord &
CUDA_VISIBLE_DEVICES= $SSL_PATH/scripts/create_unlabeled.py $ML_DATA/$PROJECT/SSL/stl10 $ML_DATA/$PROJECT/stl10-train.tfrecord $ML_DATA/$PROJECT/stl10-unlabeled.tfrecord &
CUDA_VISIBLE_DEVICES= $SSL_PATH/scripts/create_unlabeled.py $ML_DATA/$PROJECT/SSL/svhn $ML_DATA/$PROJECT/svhn-train.tfrecord &
CUDA_VISIBLE_DEVICES= $SSL_PATH/scripts/create_unlabeled.py $ML_DATA/$PROJECT/SSL/svhnx $ML_DATA/$PROJECT/svhn-train.tfrecord $ML_DATA/$PROJECT/svhn-extra.tfrecord &
wait

# Create semi-supervised subsets
for seed in 0 1 2 3 4 5; do
    for size in 40 100 250 1000 4000; do
        CUDA_VISIBLE_DEVICES= $SSL_PATH/scripts/create_split.py --seed=$seed --size=$size $ML_DATA/$PROJECT/SSL/cifar10 $ML_DATA/$PROJECT/cifar10-train.tfrecord &
        CUDA_VISIBLE_DEVICES= $SSL_PATH/scripts/create_split.py --seed=$seed --size=$size $ML_DATA/$PROJECT/SSL/svhn $ML_DATA/$PROJECT/svhn-train.tfrecord &
        CUDA_VISIBLE_DEVICES= $SSL_PATH/scripts/create_split.py --seed=$seed --size=$size $ML_DATA/$PROJECT/SSL/svhnx $ML_DATA/$PROJECT/svhn-train.tfrecord $ML_DATA/$PROJECT/svhn-extra.tfrecord &
    done
    for size in 400 1000 2500 10000; do
        CUDA_VISIBLE_DEVICES= $SSL_PATH/scripts/create_split.py --seed=$seed --size=$size $ML_DATA/$PROJECT/SSL/cifar100 $ML_DATA/$PROJECT/cifar100-train.tfrecord &
    done
    CUDA_VISIBLE_DEVICES= $SSL_PATH/scripts/create_split.py --seed=$seed --size=1000 $ML_DATA/$PROJECT/SSL/stl10 $ML_DATA/$PROJECT/stl10-train.tfrecord $ML_DATA/$PROJECT/stl10-unlabeled.tfrecord &
    wait
done
CUDA_VISIBLE_DEVICES= $SSL_PATH/scripts/create_split.py --seed=1 --size=5000 $ML_DATA/$PROJECT/stl10 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord
```

## Training

```bash
# FixMatch
python $SSL_PATH/fixmatch.py --dataset=cifar10.3@250-0 --unlabeled=cifar10 --uratio=5 --augment='CTA(sm,sm,sm)'
```

## Tensorboard

```bash
tensorboard --port 6006 --logdir_spec=experiments
```
