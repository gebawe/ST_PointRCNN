#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16                    # 1 node
#SBATCH --ntasks-per-node=1         # 36 tasks per node
#SBATCH --time=4:00:00               # time limits: 500 hour
#SBATCH --partition=gpufast 		#gpuextralong	  # gpufast
#SBATCH --gres=gpu:1
#SBATCH --output=logs/teacher_test_det_kitti_train_mf1500_w3_p16384_agum_f1_1_%j.log     # file name for stdout/stderr
# module
ml PointRCNN
## test

python ../eval_rcnn.py --cfg_file ../cfgs/test/test_kitti_train_mf1500_w3_p16384_agum_f1_1.yaml --ckpt ../output/teacher/rcnn/kitti_train_mf1500_w3_p16384_agum_f1_1/ckpt/best.pth --batch_size 1 --eval_mode rcnn --output_dir ../test/DET/teacher/kitti_train_mf1500_w3_p16384_agum_f1_1
