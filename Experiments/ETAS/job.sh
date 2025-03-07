#!/bin/bash
#SBATCH --job-name=train_ETAS
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --mem=20G
#SBATCH --account=MATH026082

# Load conda environment
source /user/work/ss15859/miniforge3/etc/profile.d/conda.sh
conda activate earthquakeNPP

# Get the config name from the command line argument
CONFIG=$1

# Run the ETAS scripts with the provided config
python invert_etas.py "$CONFIG"
python predict_etas.py "$CONFIG"

