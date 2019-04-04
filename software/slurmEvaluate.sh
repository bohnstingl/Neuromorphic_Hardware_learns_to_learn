#!/bin/sh
#SBATCH --job-name=test_parallel_scoop
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --output=ltl-out.log

pwd; hostname; date

for i in {1..12}
do
    srun python3 StateActionMaze.py
done
