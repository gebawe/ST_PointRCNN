#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16                    # 1 node
#SBATCH --ntasks-per-node=1         # 36 tasks per node
#SBATCH --time=4:00:00               # time limits: 500 hour
#SBATCH --partition=gpufast 		#gpuextralong	  # gpufast
#SBATCH --gres=gpu:1
#SBATCH --output=logs/test_det_kitti_train_mf1500_w3_p16384_agum_f1_0_%j.log     # file name for stdout/stderr
# module
ml PointRCNN
## test

python ../eval_rcnn.py --cfg_file ../cfgs/val/test_kitti_train_mf1500_w3_p16384_agum_f1_0.yaml --ckpt ../output/rcnn/kitti_train_mf1500_w3_p16384_agum_f1_0/ckpt/checkpoint_epoch_15.pth --batch_size 1 --eval_mode rcnn --output_dir test/DET/val/kitti_train_mf1500_w3_p16384_agum_f1_0
