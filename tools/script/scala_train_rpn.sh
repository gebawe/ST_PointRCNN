#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16                    # 1 node
#SBATCH --ntasks-per-node=1         # 36 tasks per node
#SBATCH --time=4:00:00               # time limits: 500 hour
#SBATCH --partition=gpufast 		#gpuextralong	  # gpufast
#SBATCH --gres=gpu:1
#SBATCH --output=logs/scala_train_rpn%j.log     # file name for stdout/stderr
ml PointRCNN #/0.0.0-fosscuda-20210b-PyTorch-1.7.1
## RPN
python ../train_rcnn.py --cfg_file ../cfgs/teacher/scala.yaml --batch_size 16 --train_mode rpn --epochs 4 --output ../output/teacher/rpn/scala_train_rpn
