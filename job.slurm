#!/bin/bash

#SBATCH -N 1 -n 1
#SBATCH --mem=4096
#SBATCH --job-name=ST
#SBATCH --time=300:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehdi.abbana-bennani@inria.fr

chmod 755 main.sh
./main.sh
