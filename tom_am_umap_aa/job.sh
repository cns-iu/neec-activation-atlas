#!/bin/bash

#SBATCH -J tom_am
#SBATCH -p gpu
#SBATCH -A legacy-projects
#SBATCH --account r00012
#SBATCH -o tom_am.txt
#SBATCH -e tom_am.err
#SBATCH --gpus-per-node v100:4
#SBATCH --time=24:00:00
#SBATCH --mem=300G

#Run your program
srun python3 activation_maximization.py