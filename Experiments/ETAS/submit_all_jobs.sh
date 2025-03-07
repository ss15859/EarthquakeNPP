#!/bin/bash

# List of config files
configs=("SaltonSea_10" "SanJac_10" "WHITE_06" "SCEDC_20" "SCEDC_25" "SCEDC_30" "ComCat_25" "Japan_25" "incomplete_California_25" "synthetic_California_25")

# Loop over all config files and submit jobs
for config in "${configs[@]}"; do
    # Create output directory
    output_dir="output_data_${config}"
    mkdir -p "$output_dir"
    
    # Submit job using SLURM
    sbatch --output="${output_dir}/job_output.out" job.sh "$config"
done