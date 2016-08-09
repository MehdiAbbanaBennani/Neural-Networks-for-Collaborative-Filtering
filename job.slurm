#!/bin/bash

#SBATCH -N 1 -n 4
#SBATCH --gres=gpu:tesla
#SBATCH --mem=4096
#SBATCH --job-name=autoencoder_basic_uniform_search_10M
#SBATCH --time=48:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehdi.abbana-bennani@inria.fr

chmod 755 main.sh
./main.sh
