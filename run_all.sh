#!/bin/bash

sed -i '1s/.*/from config.autoencoder.V_1M import Experiment/' main.py
sed -i '5s/.*/#SBATCH --job-name=AV1C/' job.slurm
sbatch job.slurm
