#!/bin/bash

#SBATCH -N 1 -n 2
#SBATCH --gres=gpu:tesla
#SBATCH --mem=4096
#SBATCH --job-name=10M_gpu_optim_2
#SBATCH --time=03:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehdi.abbana-bennani@inria.fr

chmod 755 main.sh
./main.sh
