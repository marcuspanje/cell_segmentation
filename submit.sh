#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=10000
#SBATCH --output=out/slurm-%j.out

echo start
#./evaluate.py
#python segment.py train -d cs221_dataset -c 3 -s 360 --arch drn_d_22 --batch-size 16 --epochs 50 --lr 0.001 --momentum 0.99 --step 100 --workers 4
python segment.py test -d cs221_dataset -c 3 --arch drn_d_22 --resume checkpoint_latest.pth.tar --phase test --batch-size 1

echo done
