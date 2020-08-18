!/usr/bin/env bash

train

python train_new.py --model pspp  --backbone resnet50 --dataset citys  --lr 0.01 --epochs 160 --batch-size 4