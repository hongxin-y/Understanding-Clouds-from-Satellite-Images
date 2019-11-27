#!/bin/bash
#SBATCH --job-name job1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=16g
#SBATCH --time=01-02:00:00
#SBATCH --partition=gpu
#SBATCH --mail-type=NONE
#SBATCH -o log_out

python3 main.py
