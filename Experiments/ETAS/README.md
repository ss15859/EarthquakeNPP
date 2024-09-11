# How to perform CSEP consistency tests

This notebook follows the `pyCSEP` documentation, which can be found [here](https://docs.cseptesting.org/index.html).

It assumes that daily forecasts have been simulated by running

  ```bash
  python simulate_catalog_continuation.py [dataset] [day_of_forecast]
  ```
Where `[day_of_forecast]` is the integer number of days from the beginning of the testing period and `[dataset]` is one of `ComCat_25|SaltonSea_10|SanJac_10|SCEDC_20|WHITE_06|`.

We begin by loading `pyCSEP` along with some other required libraries.


```python
import json
import datetime as dt
import matplotlib.pyplot as plt

import csep
from csep.core import regions
from csep.utils import datasets, time_utils
```

We then define the path to the output of the ETAS parameter inversion.


```python
output_dr = 'output_data_ComCat_25'
with open(output_dr+'/parameters_0.json', 'r') as f:
        inversion_output = json.load(f)
```

## Define the spatial and magnitude regions

Before we can conduct the tests, we need to define a spatial region and a set of magnitude bin edges. The magnitude bin edges # are the lower bound (inclusive) except for the last bin, which is treated as extending to infinity.


```python
# Magnitude bins properties
min_mw = inversion_output['mc']							
max_mw = 7.65
dmw = 0.1

# Create space and magnitude regions
magnitudes = regions.magnitude_bins(min_mw, max_mw, dmw)
region = regions.california_relm_region()

# Bind region information to the forecast
space_magnitude_region = regions.create_space_magnitude_region(region, magnitudes)
```

## Load the forecast data

We will load and evaluate the forecast a week following the start of the testing period.


```python
# define forecast time period
day = 7

path_to_forecasts = output_dr + '/CSEP_day_'+str(day)+'_.csv'

start_time = time_utils.strptime_to_utc_datetime(inversion_output['timewindow_end'])+ dt.timedelta(days=day)
end_time = start_time + dt.timedelta(days=1)

# load the forecast
forecast = csep.load_catalog_forecast(
path_to_forecasts,
start_time=start_time, end_time=end_time,
region=space_magnitude_region,
filter_spatial = True,
apply_filters=True
)

forecast.filters = [f'origin_time >= {forecast.start_epoch}', f'origin_time < {forecast.end_epoch}', f'magnitude >= {forecast.min_magnitude}']
```

## Plot expected event counts

We can plot the expected event counts in each bin to visualise the day's forecast.


```python
_ = forecast.get_expected_rates(verbose=False, )
ax = forecast.expected_rates.plot(plot_args={'clim': [-3.5, 0]}, show=True)
plt.show()
```


    
![png](README_files/README_13_0.png)
    


## Define the observed catalog

We load the observed data and filter it to be compared with the daily forecast.


```python
############# define observed catalog
cat = csep.load_catalog(inversion_output['fn_catalog'])		
cat.name = 'ComCat'

cat = cat.filter_spatial(forecast.region)
cat = cat.filter(f'magnitude >= {min_mw}')
cat = cat.filter(forecast.filters)	

fn_result = output_dr+'/tests_CSEP_day_'+str(day)+'_'	
```

## Number test

Aim: The number test aims to evaluate if the number of observed events is consistent with the forecast.

Method: The observed statistic in this case is given by $N_{obs}$, which is simply the number of events in the observed catalog. To build the test distribution from the forecast, we simply count the number of events in each simulated catalog $N_j$ (for $j=1,\dots,J$ repeat simulations).

We can then evaluate the probabilities of at least and at most N events, in this case using the empirical cumlative distribution function of $F_N$:

$$\delta_1 = \mathbb{P}(N_j \geq N_{obs}) = 1 - F_N(N_{obs}-1)$$

and

$$\delta_2 = \mathbb{P}(N_j \leq N_{obs}) = F_N(N_{obs})$$

This can be performed in `pyCSEP` like this.


```python
number_test_result = csep.core.catalog_evaluations.number_test(forecast, cat,verbose=False)
ax = number_test_result.plot(show=True)
result_json = json.dumps(number_test_result.to_dict())
with open(fn_result + "number.json", "w") as f:
    f.write(result_json)
plt.show()
```


    
![png](README_files/README_21_0.png)
    


## Spatial test

Aim: The spatial test again aims to isolate the spatial component of the forecast and test the consistency of spatial rates with observed events.

Method We perform the spatial test using the expected earthquake rates $\lambda_s$ (calculated and plotted above). The observed spatial test statistic is calculated as

$$ S_{obs} = \Bigg[\sum_{i=1}^{N_{obs}}\log \lambda_s(k_i) \Bigg]N_{obs}^{-1} $$

where $\lambda_s(k_i)$ is the normalised approximate rate density in the $k^{th}$ cell corresponding to the $i^{th}$ event in the observed catalog.

Similarly, we define the test distribution using

$$ S_c = \Bigg[\sum_{i=1}^{N_{j}}\log \lambda_s(k_{ij}) \Bigg]N_{j}^{-1}; \ \ j=1,\dots,J $$ 

Finally, the quantile score for the spatial test is determined by once again comparing the observed and test distribution statistics:

$$ \gamma_s = F_s(S_{obs}) = \mathbb{P}(S_j \leq S_{obs}) $$ 

This can be performed in `pyCSEP` like this.


```python
spatial_test_result = csep.core.catalog_evaluations.spatial_test(forecast, cat, verbose=False)
ax = spatial_test_result.plot(show=True)
with open(fn_result + "spatial.json", "w") as f:
    f.write(result_json)
plt.show()
```


    
![png](README_files/README_25_0.png)
    


## Magnitude test

Aim: The magnitude test aims to test the consistency of the observed frequency-magnitude distribution with that in the simulated catalogs that make up the forecast.

Method: We first define the union catalog $\Lambda_U$ as the union of all simulated catalogs in the forecast. Formally:

$$ \Lambda_U = { \lambda_1 \cup \lambda_2 \cup ... \cup \lambda_j } $$

so that the union catalog contains all events across all simulated catalogs for a total of $N_U = \sum_{j=1}^{J} \big{|}\lambda_j\big{|}$ events. We then compute the following histograms discretised to the magnitude range and magnitude step size (specified earlier for pyCSEP): 1. the histogram of the union catalog magnitudes $\Lambda_U^{(m)}$ 2. Histograms of magnitudes in each of the individual simulated catalogs $\lambda_j^{(m)}$ 3. the histogram of the observed catalog magnitudes $\Omega^{(m)}$.

The histograms are normalized so that the total number of events across all bins is equal to the observed number. The observed statistic is then calculated as the sum of squared logarithmic residuals between the normalised observed magnitudes and the union histograms. This statistic is related to the Kramer von-Mises statistic.

$$ d_{obs}= \sum_{k}\Bigg(\log\Bigg[\frac{N_{obs}}{N_U} \Lambda_U^{(m)}(k) + 1\Bigg]- \log\Big[\Omega^{(m)}(k) + 1\Big]\Bigg)^2$$

where $\Lambda_U^{(m)}(k)$ and $\Omega^{(m)}(k)$ represent the count in the $k^{th}$ bin of the magnitude-frequency distribution in the union and observed catalogs respectively. We add unity to each bin to avoid $\log(0)$. We then build the test distribution from the catalogs in $\boldsymbol{\Lambda}$:

$$ D_j = \sum_{k}\Bigg(\log\Bigg[\frac{N_{obs}}{N_U} \Lambda_U^{(m)}(k) + 1\Bigg]- \log\Bigg[\frac{N_{obs}}{N_j}\Lambda_j^{(m)}(k) + 1\Bigg]\Bigg)^2; j= 1...J$$

where $\lambda_j^{(m)}(k)$ represents the count in the $k^{th}$ bin of the magnitude-frequency distribution of the $j^{th}$ catalog.

The quantile score can then be calculated using the empirical CDF such that

$$\gamma_m = F_D(d_{obs})= P(D_j \leq d_{obs})$$

This can be performed in `pyCSEP` like this.


```python
magnitude_test_result = csep.core.catalog_evaluations.magnitude_test(forecast, cat,verbose = False)
ax = magnitude_test_result.plot(show=True)
result_json = json.dumps(magnitude_test_result.to_dict())
with open(fn_result + "magnitude.json", "w") as f:
    f.write(result_json)
plt.show()
```


    
![png](README_files/README_29_0.png)
    

