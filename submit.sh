#!/bin/bash
#SBATCH --job-name=train1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

echo hello
./train1.py
