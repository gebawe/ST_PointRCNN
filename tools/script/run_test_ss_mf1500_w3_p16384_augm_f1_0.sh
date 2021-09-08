#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16                    # 1 node
#SBATCH --ntasks-per-node=1         # 36 tasks per node
#SBATCH --time=24:00:00               # time limits: 500 hour
#SBATCH --partition=gpu 		#gpuextralong	  # gpufast
#SBATCH --gres=gpu:1
#SBATCH --output=logs/test_ss_kitti_train_mf1500_w3_p16384_agum_f1_0_%j.log     # file name for stdout/stderr
# module
ml PointRCNN
## test

python ../eval_rcnn.py --cfg_file ../cfgs/val/test_kitti_train_mf1500_w3_p16384_agum_f1_0.yaml --ckpt ../output/rpn/kitti_train_mf1500_w3_p16384_agum_f1_0/ckpt/best.pth --batch_size 1 --eval_mode rpn --output_dir test/SS/val/val/kitti_train_mf1500_w3_p16384_agum_f1_0
