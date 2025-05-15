import os
import glob
import argparse
from datetime import datetime, timedelta
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import csep
import csep
from csep.core import regions
from csep.utils import datasets, time_utils

import sys
import json
import cartopy




def get_dataset_config(dataset_name):
	"""
	Get configuration for each dataset including magnitude cutoff, region, and test_nll_start
	
	Parameters:
	-----------
	dataset_name : str
		Name of the dataset (ComCat, WHITE, SCEDC, SanJac, SaltonSea)
		
	Returns:
	--------
	Mc : float
		Magnitude cutoff
	region : csep.core.regions object
		Region object for spatial tests
	test_nll_start : str
		Start datetime for NLL testing as a string
	"""
	if dataset_name == 'ComCat' or dataset_name == 'ComCat_25':
		Mc = 2.5
		region = regions.california_relm_region()
		test_nll_start = '2007-01-01 00:00:00'
	
	elif dataset_name == 'SaltonSea' or dataset_name == 'SaltonSea_10':
		Mc = 1.0
		region = csep.core.regions.CartesianGrid2D.from_origins(
			np.array(list(product(
				np.arange(-116, -115, 0.01), np.arange(32.5, 33.3, 0.01)
			)))
		)
		test_nll_start = '2016-01-01 00:00:00'
	
	elif dataset_name == 'SanJac' or dataset_name == 'SanJac_10':
		Mc = 1.0
		region = csep.core.regions.CartesianGrid2D.from_origins(
			np.array(list(product(
				np.arange(-117, -116, 0.01), np.arange(33, 34, 0.01)
			)))
		)
		test_nll_start = '2016-01-01 00:00:00'
	
	elif dataset_name == 'WHITE' or dataset_name == 'WHITE_06':
		Mc = 0.6
		region = csep.core.regions.CartesianGrid2D.from_origins(
			np.array(list(product(
				np.arange(-117.133, -116, 0.01), np.arange(33, 34, 0.01)
			)))
		)
		test_nll_start = '2017-01-01 00:00:00'
	
	elif dataset_name == 'SCEDC' or dataset_name == 'SCEDC_20':
		Mc = 2.0
		region = csep.core.regions.CartesianGrid2D.from_origins(
			np.array(list(product(
				np.arange(-122, -114, 0.05), np.arange(32, 37, 0.05)
			)))
		)
		test_nll_start = '2014-01-01 00:00:00'

	elif dataset_name == 'China_20':
		Mc = 2.0
		region = csep.core.regions.CartesianGrid2D.from_origins(
			np.array(list(product(
				np.arange(-122, -114, 0.05), np.arange(32, 37, 0.05)
			)))
		)
		test_nll_start = '2013-01-01 00:00:00'

	elif dataset_name == 'China_24':
		Mc = 2.4
		region = csep.core.regions.CartesianGrid2D.from_origins(
			np.array(list(product(
				np.arange(-122, -114, 0.05), np.arange(32, 37, 0.05)
			)))
		)
		test_nll_start = '2013-01-01 00:00:00'
	
	else:
		raise ValueError(f"Unknown dataset: {dataset_name}")
	
	return Mc, region, test_nll_start


def run_tests(forecast_path, catalog_path, plots_path, test_results_path, test_day, model_name, min_magnitude, region, test_nll_start):
	"""
	Run PyCSEP tests on earthquake forecasts
	"""

	# Convert test_nll_start to datetime
	test_nll_start_dt = time_utils.strptime_to_utc_datetime(test_nll_start)

	# Create space-magnitude region
	min_mw = min_magnitude
	max_mw = 7.65
	dmw = 0.1
	magnitudes = regions.magnitude_bins(min_mw, max_mw, dmw)
	space_magnitude_region = regions.create_space_magnitude_region(region, magnitudes)

	# Define forecast time period
	path_to_forecast_day = forecast_path + '/CSEP_day_' + str(test_day) + '.csv'

	### Ensure forecast file is in CSEP format
	df = pd.read_csv(path_to_forecast_day)

	# sort by catalog_id then time_string
	df2 = df.sort_values(by=['catalog_id','time_string'])
	if not df2.equals(df):
		print('catalog_id and time_string are not sorted')
		df2.to_csv(path_to_forecast_day, index=False)
	else:
		print('catalog_id and time_string are sorted')


	if 'event_id' not in df.columns:
		df['event_id'] = range(1, len(df)+1)
		df.to_csv(path_to_forecast_day, index=False)

	


	start_time = test_nll_start_dt + timedelta(days=test_day)
	end_time = start_time + timedelta(days=1)

	forecast = csep.load_catalog_forecast(
		path_to_forecast_day,
		start_time=start_time, end_time=end_time,
		region=space_magnitude_region,
		filter_spatial=True,
		apply_filters=True
	)

	forecast.filters = [f'origin_time >= {forecast.start_epoch}', f'origin_time < {forecast.end_epoch}', f'magnitude >= {forecast.min_magnitude}']
	expected_rates = forecast.get_expected_rates(verbose=True)

	args_forecast = {'title': 'Landers aftershock forecast',
					 'grid_labels': True,
					 'basemap': 'ESRI_imagery',
					 'cmap': 'rainbow',
					 'alpha_exp': 0.5,
					 'projection': cartopy.crs.Mercator(),
					 'clim': [-3.5, 0]}

	# Define observed catalog
	cat = csep.load_catalog(catalog_path)
	# make a string of the date of the start time
	start_time = start_time.strftime('%Y-%m-%d')
	cat.name = model_name + f' {start_time}'

	cat = cat.filter_spatial(forecast.region)
	cat = cat.filter(f'magnitude >= {min_mw}')
	cat = cat.filter(forecast.filters)

	args_catalog = {'basemap': 'ESRI_terrain',
					'markercolor': 'grey',
					'markersize': 1,
					'title_size': 20,
					'grid':True,
					'grid_labels': True,
					'grid_fontsize': 10}
	
	if cat.event_count > 0:
		ax_1 = expected_rates.plot(plot_args=args_forecast)
		ax_2 = cat.plot(ax=ax_1, plot_args=args_catalog)
		plt.savefig(plots_path+ f'/day_{test_day}.png')


	######################### TESTS
	#### Number Test
	number_test_result = csep.core.catalog_evaluations.number_test(forecast, cat)
	ax = number_test_result.plot(show=False,plot_args ={'title_fontsize':16})
	plt.savefig(plots_path+ f'/day_{test_day}_number.png')
	result_json = json.dumps(number_test_result.to_dict())
	with open(test_results_path + f"/tests_CSEP_day_{test_day}_number.json", "w") as f:
		f.write(result_json)

	#### Spatial Test
	if cat.event_count>0:
		spatial_test_result = csep.core.catalog_evaluations.spatial_test(forecast, cat)
		ax = spatial_test_result.plot(show=False,plot_args ={'title_fontsize':16})
		plt.savefig(plots_path+ f'/day_{test_day}_spatial.png')
		result_json = json.dumps(spatial_test_result.to_dict())
		with open(test_results_path + f"/tests_CSEP_day_{test_day}_spatial.json", "w") as f:
			f.write(result_json)

	#### Pseudolikelihood Test
	if cat.event_count>0:
		try :
			pseudolikelihood_test_result = csep.core.catalog_evaluations.pseudolikelihood_test(forecast, cat)
			ax = pseudolikelihood_test_result.plot(show=False,plot_args ={'title_fontsize':16})
			plt.savefig(plots_path+ f'/day_{test_day}_pseudolikelihood.png')
			result_json = json.dumps(pseudolikelihood_test_result.to_dict())
			with open(test_results_path + f"/tests_CSEP_day_{test_day}_pseudolikelihood.json", "w") as f:
				f.write(result_json)
		except:
			pass

	#### Magnitude Test
	if cat.event_count>0:
		magnitude_test_result = csep.core.catalog_evaluations.magnitude_test(forecast, cat)
		ax = magnitude_test_result.plot(show=False,plot_args ={'title_fontsize':16})
		plt.savefig(plots_path+ f'/day_{test_day}_magnitude.png')
		result_json = json.dumps(magnitude_test_result.to_dict())
		with open(test_results_path + f"/tests_CSEP_day_{test_day}_magnitude.json", "w") as f:
			f.write(result_json)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run PyCSEP tests on earthquake forecasts')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['SMASH', 'ETAS', 'DSTPP'], 
                        help='Model name (SMASH, ETAS, DSTPP)')
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['ComCat', 'WHITE', 'SCEDC', 'SanJac', 'SaltonSea'],
                        help='Dataset name')
    parser.add_argument('--test_day', type=int, default=0,
                        help='Day since the start of test period')
    parser.add_argument('--HPC', type=bool, default=False,
                        help='Run on HPC')
    
    args = parser.parse_args()

    Mc, region, test_nll_start = get_dataset_config(args.dataset)
    
    # Format Mc value
    Mc_str = f"{int(Mc * 10):02d}"
    
    # Set default paths based on model and dataset if not provided
    if args.HPC:
        forecast_path = '/user/work/ss15859/'
    else:
        forecast_path = '../'

    if args.model == 'SMASH':
        args.forecast_dir = forecast_path + f'SMASH_daily_forecasts/{args.dataset}'
        args.plots_path = forecast_path + f'SMASH_daily_forecasts/{args.dataset}/plots'
        args.test_results_path = forecast_path + f'SMASH_daily_forecasts/{args.dataset}/test_results'
    elif args.model == 'ETAS':
        args.forecast_dir = forecast_path + f'/ETAS/output_data_{args.dataset}_{Mc_str}/forecasts'
        args.plots_path = forecast_path + f'/ETAS/output_data_{args.dataset}_{Mc_str}/plots'
        args.test_results_path = forecast_path + f'/ETAS/output_data_{args.dataset}_{Mc_str}/test_results'
    elif args.model == 'DSTPP':
        args.forecast_dir = forecast_path + f'DSTPP_daily_forecasts/{args.dataset}'
        args.plots_path = forecast_path + f'DSTPP_daily_forecasts/{args.dataset}/plots'
        args.test_results_path = forecast_path + f'DSTPP_daily_forecasts/{args.dataset}/test_results'

    # if plots path doesn't exist, create it
    if not os.path.exists(args.plots_path):
        os.makedirs(args.plots_path)
    if not os.path.exists(args.test_results_path):
        os.makedirs(args.test_results_path)

    args.catalog_path = f'../ETAS/output_data_{args.dataset}_{Mc_str}/CSEP_format_catalog.csv'
    
    print(f"Running PyCSEP tests for {args.model} on {args.dataset}")
    print(f"Forecast directory: {args.forecast_dir}")
    print(f"Catalog file: {args.catalog_path}")
    print(f"Test Day: {args.test_day}")
    
    # Run tests
    results = run_tests(
        forecast_path=args.forecast_dir,
        catalog_path=args.catalog_path,
        plots_path=args.plots_path,
        test_results_path=args.test_results_path,
        test_day=args.test_day,
        model_name=f"{args.dataset} {args.model}",
        min_magnitude=Mc,
        region=region,
        test_nll_start=test_nll_start
    )
    
    print("Tests completed")

if __name__ == "__main__":
    main()
