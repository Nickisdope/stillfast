#!/bin/bash

#SBATCH --output="/srv/beegfs02/scratch/gaze_pred/data/xiang/stillfast_log/%j.out"
#SBATCH --gres=gpu:4
#SBATCH --constraint='titan_xp|geforce_gtx_titan_x'
#SBATCH --job-name=stillfast

source /scratch_net/snapo/xianliu/conda/etc/profile.d/conda.sh
conda activate stillfast

nvidia-smi

python -u main.py --cfg configs/sta/STILL_FAST_R50_X3DM_EGO4D_v1_modified.yaml --train --exp trial 