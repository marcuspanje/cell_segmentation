#!/bin/bash

#SBATCH --job-name=train1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

echo hello
python segment_preprocess.py train -d cs221_dataset -c 3 -s 360 --arch drn_d_22 --batch-size 16 --epochs 2 --lr 0.001 --momentum 0.99 --step 100
