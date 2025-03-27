#!/bin/bash
# filepath: /home/ss15859/Documents/EarthquakeNPP/Experiments/ETAS/move_files.sh

# Define the datasets array properly
# datasets=("ComCat" "SaltonSea" "SanJac" "WHITE" "SCEDC")
datasets=("SCEDC")

# Loop through each dataset
for dataset in "${datasets[@]}"
do
    for day in {96..110}
    do
        python run_pycsep_tests.py --model ETAS  --dataset ${dataset} --test_day ${day}
    done
done
    