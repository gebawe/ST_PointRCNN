#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16                    # 1 node
#SBATCH --ntasks-per-node=1         # 36 tasks per node
#SBATCH --time=4:00:00               # time limits: 500 hour
#SBATCH --partition=gpufast 		#gpuextralong	  # gpufast
#SBATCH --gres=gpu:1
#SBATCH --output=logs/teacher_rcnn_kitti_train_mf1500_w3_p16384_agum_f1_1_%j.log     # file name for stdout/stderr
ml PointRCNN
## RCNN
python ../train_rcnn.py --cfg_file ../cfgs/teacher/kitti_train_mf1500_w3_p16384_agum_f1_1.yaml --batch_size 16 --train_mode rcnn --epochs 70 --ckpt_save_interval 1 --rpn_ckpt ../output/rpn/kitti_train_mf1500_w3_p16384_agum_f1_1/ckpt/best.pth --output ../output/teacher/rcnn/kitti_train_mf1500_w3_p16384_agum_f1_1

