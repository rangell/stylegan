#!/bin/bash

srun -p 1080ti-short --gres=gpu:4 --mem=30000 run_load_encoder.sh
