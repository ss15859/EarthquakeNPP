#!/bin/bash

# Create the conda environment
conda env create -f environment.yml;

# Activate earthquakeNPP
conda activate earthquakeNPP;

# Use pip to install the remaining packages
pip install -r requirements.txt;