#!/bin/bash

sed -i '1s/.*/from config.autoencoder.U_1M import Experiment/' main.py
sed -i '5s/.*/#SBATCH --job-name=AU1C/' job.slurm
sbatch job.slurm

sed -i '1s/.*/from config.autoencoder.V_1M import Experiment/' main.py
sed -i '5s/.*/#SBATCH --job-name=AV1C/' job.slurm
sbatch job.slurm

sed -i '1s/.*/from config.autoencoder.U_10M import Experiment/' main.py
sed -i '5s/.*/#SBATCH --job-name=AU10C/' job.slurm
sbatch job.slurm

sed -i '1s/.*/from config.autoencoder.V_10M import Experiment/' main.py
sed -i '5s/.*/#SBATCH --job-name=AV10C/' job.slurm
sbatch job.slurm

sed -i '1s/.*/from config.stability.U_1M import Experiment/' main.py
sed -i '5s/.*/#SBATCH --job-name=SU1C/' job.slurm
sbatch job.slurm

sed -i '1s/.*/from config.stability.V_1M import Experiment/' main.py
sed -i '5s/.*/#SBATCH --job-name=SV1C/' job.slurm
sbatch job.slurm

sed -i '1s/.*/from config.stability.U_10M import Experiment/' main.py
sed -i '5s/.*/#SBATCH --job-name=SU10C/' job.slurm
sbatch job.slurm

sed -i '1s/.*/from config.stability.V_10M import Experiment/' main.py
sed -i '5s/.*/#SBATCH --job-name=SV10C/' job.slurm
sbatch job.slurm