#!/bin/bash

#SBATCH -N 1 -n 2
#SBATCH --mem=8192
#SBATCH --job-name=EU1G
#SBATCH --time=300:00:00

#SBATCH --gres=gpu:K80

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehdi.abbana-bennani@inria.fr

chmod 755 main.sh
./main.sh
