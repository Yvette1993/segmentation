!/usr/bin/env bash

train

python train_new.py --model danet   --backbone resnet50 --dataset citys  --lr 0.01 --epochs 80 --batch-size 4

# test
# export NGPUS=2
# python -m torch.distributed.launch --nproc_per_node=$NGPUS  eval.py --model danet --backbone resnet50 --dataset citys --resume ./torch/models/danet_resnet50_citys_best_model.pth 
# python eval.py --model danet --backbone resnet50 --dataset citys --resume ./torch/models/danet_resnet50_citys_best_model.pth  --batch-size 1
# inf
# python demo.py --model danet  --input-pic ./datasets/test.jpg

