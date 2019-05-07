#!/bin/bash

source ~/.bashrc

module load cuda90 cudnn/7.3-cuda_9.0

source activate stylegan

python ../stylegan/pretrained_example.py
