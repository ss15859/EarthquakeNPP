#!/bin/bash
#PBS -N romeq
#PBS -q romeq
#PBS -l select=1:ncpus=1
#PBS -l walltime=200:00:00

cd $PBS_O_WORKDIR

module list

pwd

echo $CUDA_VISIBLE_DEVICES

# Load local python environment
source activate earthquakeNPP

python invert_etas.py $CONFIG

