#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out/slurm-%j.out

echo start
#./train.py
./evaluate.py
echo done
