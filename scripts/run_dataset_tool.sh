#!/bin/bash

source ~/.bashrc

module load cuda90 cudnn/7.3-cuda_9.0

source activate stylegan

python ../dataset_tool.py create_from_images /mnt/nfs/scratch1/sbrockman/datasets/vggface2 /mnt/nfs/scratch1/sbrockman/vggface2/whole_dataset
