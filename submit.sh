#!/bin/bash
#SBATCH --job-name=train1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

echo start
./train.py
echo done
