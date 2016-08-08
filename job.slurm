#!/bin/bash

#SBATCH -N 1 -n 2
#SBATCH --gres=gpu:K80
#SBATCH --mem=4096
#SBATCH --job-name=autoencoder_stability_search_fixed_first_learning
#SBATCH --time=48:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehdi.abbana-bennani@inria.fr

chmod 755 main.sh
./main.sh
