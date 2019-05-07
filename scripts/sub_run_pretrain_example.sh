#!/bin/bash

srun -p titanx-short --gres=gpu:1 run_pretrain_example.sh
