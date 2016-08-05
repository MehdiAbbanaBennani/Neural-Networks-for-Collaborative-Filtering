#!/bin/bash

#SBATCH -N 1 -n 1
#SBATCH --time=14:00:00
#SBATCH --job-name=one_node_one_cpu
#SBATCH --mem=4096
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehdi.abbana-bennani@inria.fr

chmod 755 main.sh
./main.sh
