#!/bin/sh
#SBATCH --job-name=test_parallel_scoop
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=12
#SBATCH --output=ltl-out.log

pwd; hostname; date

python3 -m scoop -n $((12*2)) ltl-SNN-gd.py

