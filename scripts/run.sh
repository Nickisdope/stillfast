# distributed training with 2 titan x
srun --time 30 --partition=gpu.debug --gres=gpu:4 --constraint='geforce_gtx_titan_x' --pty bash -i
srun --time 30 --partition=gpu.debug --gres=gpu:2 --constraint='geforce_gtx_titan_x' --pty bash -i
srun --time 60 --partition=gpu.debug --gres=gpu:1 --constraint='geforce_gtx_titan_x' --pty bash -i

# debug training and validation loop
python main.py --cfg configs/sta/STILL_FAST_R50_X3DM_EGO4D_v1_debug.yaml --train --fast_dev_run