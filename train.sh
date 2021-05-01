#!/bin/bash
shift
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -m torch.distributed.launch --nproc_per_node=8 main.py >./logs/EfficientNetV2S2.out 2>&1


