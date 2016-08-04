#!/bin/bash                                                                                                               
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=14:00:00
#SBATCH --job-name=autoencoder_experiment
#SBATCH --mem=4096M
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehdi.abbana-bennani@inria.fr

chmod 755 main.sh
./main.sh
