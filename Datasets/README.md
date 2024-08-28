# EarthquakeNPP Datasets

Each subdirectory provides information on the construction of each EarthquakeNPP Dataset, including details about the source, as well as how we preprocess each dataset for our platform.

Each preprocessed dataset is also included within each subdirectory.

#### Official EarthquakeNPP datasets

The following datasets are considered to be the official EarthquakeNPP datasets.

`ComCat_25`, `SCEDC_20`, `SCEDC_25` and `SCEDC_30` represent datasets currently being used for operational/realtime forecasting in California by the [USGS](https://earthquake.usgs.gov/earthquakes/eventpage/ci40699207/executive). Different magnitude thresholds of the `SCEDC` catalog allow us to assess how magnitude threshold affects forecasting performance.

`QTM_SaltonSea_10`, `QTM_SanJac_10` and `WHITE_06` represent modern high resolution catalogs that contain very low magnitude earthquakes. This allows us to assess whether NPPs can exploit any additional signal that is found within the low magnitude events.

Our recommendation would be to use `ComCat_25` and `WHITE_06` as datasets for general benchmarking of NPPs in multi-domain experiments. 

#### Additional datasets

We include 3 additional datasets in the benchmarking experiment for various reasons:

`Japan_Deprecated`:

We include this dataset to provide a continuity between our platform and previous NPP benchmarking on earthquakes. However, since this dataset does not represent complete data of the region, it is not considered to be an official EarthquakeNPP benchmark. A future version of EarthquakeNPP will contain extensive datasets of Japan in a similar manner to our current focus on California.

`ETAS_25`, `ETAS_incomplete_25`:

We include these synthetic ETAS generated datasets to artificially recreate the data missingness often present following large magnitude events. This allows us to test whether NPPs are inhibited by data missingness to the same extent that ETAS is.

## Data Format

The datasets follow the `.csv` format:

| id    | time                    | longitude  | latitude  | magnitude | x               | y               |
|-------|-------------------------|------------|-----------|-----------|-----------------|-----------------|
| 12036 | 1971-01-01 20:36:17.720 | -119.4328333 | 33.8986667 | 2.91      | -221.8284296413594 | -52.446501577338154 |
| &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |


### Field Descriptions

- **id**: The event identification number is a unique number assigned to every event in the earthquake catalog.

- **time**: The time when the event occurred, represented in a human-readable format: YYYY-MM-DD HH:MM:SS.sss.

  We indicate the date and time when the earthquake initiates rupture, which is known as the "origin" time. Note that large earthquakes can continue rupturing for many tens of seconds. We provide time in UTC (Coordinated Universal Time). Seismologists use UTC to avoid confusion caused by local time zones and daylight savings time. On the individual event pages, times are also provided for the time at the epicenter and your local time based on the time your computer is set.

- **longitude**: Decimal degrees longitude. Negative values indicate western longitudes.

  An earthquake begins to rupture at a hypocenter, defined by a position on the surface of the earth (epicenter) and a depth below this point (focal depth). We provide the coordinates of the epicenter in units of latitude and longitude. The longitude is the number of degrees east (E) or west (W) of the prime meridian, which runs through Greenwich, England. Coordinates are given in the WGS84 reference frame. The position uncertainty of the hypocenter location varies from about 100 m horizontally and 300 meters vertically for the best-located events to tens of kilometers for global events.

- **latitude**: Decimal degrees latitude. Negative values indicate southern latitudes.

  Similar to longitude, the latitude represents the number of degrees north (N) or south (S) of the equator, varying from 0 at the equator to 90 at the poles. Coordinates are provided in the WGS84 reference frame with position uncertainty similar to longitude.

- **magnitude**: The magnitude of the event.

  The magnitude reported is considered official for this earthquake by each data center. Earthquake magnitude is a measure of the size of an earthquake at its source, expressed logarithmically. The amplitude of seismic waves increases by approximately 10 times for each unit increase in magnitude.

- **x, y**: Azimuthal equidistant projection of coordinates (latitude, longitude).

  Since ETAS uses the great-circle distance between two points on a sphere (km), we project the coordinates of the events into a space where inter-event distances are in kilometers using the Azimuthal equidistant projection. NPPs should use these columns as the spatial locations of each earthquake.

## Dataset Partitioning

For use in the benchmarking experiment, each catalog is partitioned for training, validation, and testing using the following dates:

| Catalog      | Mc  | Auxiliary Start | Training Start | Validation Start | Testing Start | Testing End | No. Training Events | No. Testing Events |
|--------------|-----|-----------------|----------------|------------------|---------------|-------------|---------------------|--------------------|
| Com_Cat      | 2.5 | 1971-01-01      | 1981-01-01     | 1998-01-01       | 2007-01-01    | 2020-01-17  | 79,037              | 23,059             |
| SCEDC_2.0    | 2.0 | 1981-01-01      | 1985-01-01     | 2005-01-01       | 2014-01-01    | 2020-01-01  | 128,265             | 14,351             |
| SCEDC_2.5    | 2.5 | 1981-01-01      | 1985-01-01     | 2005-01-01       | 2014-01-01    | 2020-01-01  | 43,221              | 5,466              |
| SCEDC_3.0    | 3.0 | 1981-01-01      | 1985-01-01     | 2005-01-01       | 2014-01-01    | 2020-01-01  | 12,426              | 2,065              |
| QTM_San_Jac  | 1.0 | 2008-01-01      | 2009-01-01     | 2014-01-01       | 2016-01-01    | 2018-01-01  | 18,664              | 4,837              |
| QTM_Salton_Sea | 1.0 | 2008-01-01  | 2009-01-01     | 2014-01-01       | 2016-01-01    | 2018-01-01  | 44,042              | 4,393              |
| WHITE        | 0.6 | 2008-01-01      | 2009-01-01     | 2014-01-01       | 2017-01-01    | 2021-01-01  | 38,556              | 26,914             |
| ETAS         | 1.0 | 1971-01-01      | 1981-01-01     | 1998-01-01       | 2007-01-01    | 2020-01-17  | 117,550             | 43,327             |
| ETAS_mc(t)   | 1.0 | 1971-01-01      | 1981-01-01     | 1998-01-01       | 2007-01-01    | 2020-01-17  | 115,115             | 42,932             |
| Japan_Deprecated   | 2.5 | 1990-01-01      | 1992-01-01     | 2007-01-01       | 2011-01-01    | 2020-01-01  | 22,213             | 15,368             |

