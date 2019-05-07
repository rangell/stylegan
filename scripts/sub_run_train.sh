#!/bin/bash

srun -p 1080ti-long --gres=gpu:4 --mem=30000 run_train.sh
