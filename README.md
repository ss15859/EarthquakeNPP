# EarthquakeNPP

## Description
EarthquakeNPP is an expanding collection of benchmark datasets designed to facilitate testing of Neural Point Processes (NPP) on earthquake data. The datasets are accompanied by an implementation of the Epidemic-Type Aftershock Sequence (ETAS) model, currently considered the benchmark forecasting model in the seismology community. These datasets cover various target regions within California, spanning from 1971 to 2021, and are generated using different methodologies. Derived from publicly available raw data, these datasets undergo processing and configuration to support forecasting experiments relevant to stakeholders in seismology.

## Setup

To get started with EarthquakeNPP:

1. Clone the repository and its submodules:
   ```bash
   git clone --recurse-submodules https://github.com/ss15859/EarthquakeNPP.git
   ````
2. Navigate to the cloned directory:
   ```bash
    cd EarthquakeNPP
    ```
3. Create the conda environment:
    ```bash
    conda env create -f environment.yml
    ```
4. Activate the conda environment:
    ```bash
    conda activate earthquakeNPP
    ```

## Licenses

This project uses code from the following repositories, each under their respective licenses:

1. AutoSTPP: MIT License
2. etas: MIT License
3. neural_stpp: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

For more details on the licenses, please refer to the LICENSE files in the respective repositories.

This repository also includes public data from the Southern California Seismic Network (SCSN) and the Southern California Earthquake Data Center (SCEDC).

### Data License

SCEDC herby grants the non-exclusive, royalty-free, non-transferable, worldwide right and license to use, reproduce, and publicly display in all media public data from the Southern California Seismic Network. Please cite the SCEDC (doi:10.7909/C3WD3xH1) and SCSN (doi:10.7914/SN/CI) for any research publications using this data.



For SCEDC and QTM catalog
License: SCEDC herby grants the non-exclusive, royalty free, non-transferable, 
  worldwide right and license to use, reproduce and publicly display in all media 
  public data from the Southern California Seismic Network. Please cite 
  the SCEDC (doi:10.7909/C3WD3xH1) and SCSN (doi:10.7914/SN/CI) for any research
  publications using this data.