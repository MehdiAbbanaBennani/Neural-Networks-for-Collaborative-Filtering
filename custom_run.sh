#!/bin/bash

sed -i '1s/.*/from config.stability.U_1M import Experiment/' main.py
sed -i '5s/.*/#SBATCH --job-name=SU1CT/' job.slurm
sbatch job.slurm