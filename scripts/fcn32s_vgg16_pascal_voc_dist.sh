#!/usr/bin/env bash

# train
python train_new.py --model fcn32s \
    --backbone vgg16 --dataset pascal_voc \
    --lr 0.01 --epochs 80 --batch-size 8