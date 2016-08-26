#!/bin/bash

sed -i '1s/.*/from config.autoencoder.V_1M import Experiment/' main.py
sed -i '5s/.*/#SBATCH --job-name=LAV1C/' job.slurm
sbatch job.slurm

sed -i '1s/.*/from config.autoencoder.U_10M import Experiment/' main.py
sed -i '5s/.*/#SBATCH --job-name=LAU10C/' job.slurm
sbatch job.slurm

sed -i '1s/.*/from config.autoencoder.V_10M import Experiment/' main.py
sed -i '5s/.*/#SBATCH --job-name=LAV10C/' job.slurm
sbatch job.slurm

sed -i '1s/.*/from config.stability.U_1M import Experiment/' main.py
sed -i '5s/.*/#SBATCH --job-name=LSU1C/' job.slurm
sbatch job.slurm

sed -i '1s/.*/from config.stability.U_10M import Experiment/' main.py
sed -i '5s/.*/#SBATCH --job-name=LSU1C/' job.slurm
sbatch job.slurm

sed -i '1s/.*/from config.stability.V_1M import Experiment/' main.py
sed -i '5s/.*/#SBATCH --job-name=LSV1C/' job.slurm
sbatch job.slurm

sed -i '1s/.*/from config.extra.U_1M import Experiment' main.py
sed -i '5s/.*/#SBATCH --job-name=LEU1C/' job.slurm
sbatch job.slurm