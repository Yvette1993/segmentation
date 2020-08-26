# !/usr/bin/env bash

# train

python train_new.py --model pspp  --backbone resnet50 --dataset citys  --lr 0.01 --epochs 160 --batch-size 4   --resume ./torch/models/pspp_resnet50_citys_best_model.pth

# test
# python eval.py --model pspp --backbone resnet50 --dataset citys --resume ./torch/models/pspp_resnet50_citys_best_model.pth  --batch-size 1  --gpu-ids 0,1

# inf
# python demo.py --model pspp_resnet50_citys  --input-pic  test_img.jpg  --resume ./torch/models/pspp_resnet50_citys_best_model.pth 