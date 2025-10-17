#!/bin/bash
num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
python -m torch.distributed.launch --nproc_per_node=${num_gpus} --use_env main.py 