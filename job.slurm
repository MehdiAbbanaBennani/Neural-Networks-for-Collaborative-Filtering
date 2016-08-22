#!/bin/bash

#SBATCH -N 1 -n 1
#SBATCH --mem=8192
#SBATCH --job-name=ALL_lundi
#SBATCH --time=300:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehdi.abbana-bennani@inria.fr

chmod 755 main.sh
./main.sh
