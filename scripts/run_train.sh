#!/bin/bash

source ~/.bashrc

module load cuda90 cudnn/7.3-cuda_9.0 nccl2

source activate stylegan

python ../train.py
