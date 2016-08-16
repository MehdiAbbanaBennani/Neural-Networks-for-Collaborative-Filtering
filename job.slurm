#!/bin/bash

#SBATCH -N 1 -n 4
#SBATCH --gres=gpu:K80
#SBATCH --mem=4096
#SBATCH --job-name=AV1MGPU
#SBATCH --time=300:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehdi.abbana-bennani@inria.fr

chmod 755 main.sh
./main.sh
