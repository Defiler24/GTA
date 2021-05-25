#!/bin/bash
shift
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -m torch.distributed.launch --nproc_per_node=8 main.py --lambdaMae 0.0 >./logs/MAE0.out 2>&1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -m torch.distributed.launch --nproc_per_node=8 main.py --lambdaMae 0.2 >./logs/MAE2.out 2>&1; CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -m torch.distributed.launch --nproc_per_node=8 main.py --lambdaMae 0.4 >./logs/MAE4.out 2>&1; 

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -m torch.distributed.launch --nproc_per_node=8 main.py --lambdaMae 0.6 >./logs/MAE6.out 2>&1; 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -m torch.distributed.launch --nproc_per_node=8 main.py >./logs/2of2glink_mae.out 2>&1



