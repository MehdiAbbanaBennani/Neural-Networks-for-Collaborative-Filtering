#!/bin/bash                                                                                                               
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --job-name=autoencoder_experiment_test
#SBATCH --mem=1024
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehdi.abbanabennani@etu.emse.fr

chmod 755 main.sh
./main.sh
