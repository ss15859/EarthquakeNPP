import numpy
import numpy as np

import csep
from csep.core import regions
from csep.utils import datasets, time_utils
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import sys
import json


if __name__ == '__main__':

	# HPC_path = '/user/work/ss15859/'
	HPC_path = ''

	if sys.argv[1]=='ComCat_25':
		dic = {'name':'ComCat_25','Mc':2.5,'path_to_cat':HPC_path +'output_data_ComCat_25/CSEP_format_catalog.csv'}
		region = regions.california_relm_region()

	if sys.argv[1]=='SaltonSea_10':
		dic = {'name':'SaltonSea_10','Mc':1.0,'path_to_cat':HPC_path +'output_data_SaltonSea_10/CSEP_format_catalog.csv'}
		region = csep.core.regions.CartesianGrid2D.from_origins(                   
			np.array(list(product(
				np.arange(-116, -115, 0.01), np.arange(32.5, 33.3, 0.01)
			)))
		)

	if sys.argv[1]=='SanJac_10':	
		dic = {'name':'SanJac_10','Mc':1.0,'path_to_cat':HPC_path +'output_data_SanJac_10/CSEP_format_catalog.csv'}
		region = csep.core.regions.CartesianGrid2D.from_origins(                   
			np.array(list(product(
				np.arange(-117, -116, 0.01), np.arange(33, 34, 0.01)
			)))
		)

	if sys.argv[1]== 'WHITE_06':
		dic = {'name':'WHITE_06','Mc':0.6,'path_to_cat':HPC_path +'output_data_WHITE_06/CSEP_format_catalog.csv'}
		region = csep.core.regions.CartesianGrid2D.from_origins(                   
			np.array(list(product(
				np.arange(-117.133, -116, 0.01), np.arange(33, 34, 0.01)
			)))
		)

	if sys.argv[1]== 'SCEDC_30':
		dic = {'name':'SCEDC_30','Mc':3.0,'path_to_cat':HPC_path +'output_data_SCEDC_30/CSEP_format_catalog.csv'}
		region = csep.core.regions.CartesianGrid2D.from_origins(                   
			np.array(list(product(
				np.arange(-122, -114, 0.05), np.arange(32,37, 0.05)
			)))
		)

	if sys.argv[1]== 'SCEDC_25':
		dic = {'name':'SCEDC_25','Mc':2.5,'path_to_cat':HPC_path +'output_data_SCEDC_25/CSEP_format_catalog.csv'}
		region = csep.core.regions.CartesianGrid2D.from_origins(                   
			np.array(list(product(
				np.arange(-122, -114, 0.05), np.arange(32,37, 0.05)
			)))
		)

	if sys.argv[1]== 'SCEDC_20':
		dic = {'name':'SCEDC_20','Mc':2.0,'path_to_cat':HPC_path +'output_data_SCEDC_20/CSEP_format_catalog.csv'}
		region = csep.core.regions.CartesianGrid2D.from_origins(                   
			np.array(list(product(
				np.arange(-122, -114, 0.05), np.arange(32,37, 0.05)
			)))
		)



	home_path = 'output_data_'+dic['name']
	work_path = HPC_path +'output_data_'+dic['name']
	parameter_dr = home_path+'/parameters_0.json'

	with open(parameter_dr, 'r') as f:
	        inversion_output = json.load(f)


	# Magnitude bins properties
	min_mw = dic['Mc']															
	max_mw = 7.65
	dmw = 0.1
	magnitudes = regions.magnitude_bins(min_mw, max_mw, dmw)

	# Define space-magnitude region
	space_magnitude_region = regions.create_space_magnitude_region(region, magnitudes)


	# define forecast time period
	day = int(sys.argv[2])

	path_to_forecasts = work_path+'/CSEP_day_'+str(day)+'_.csv'

	start_time = time_utils.strptime_to_utc_datetime(inversion_output['timewindow_end'])+ dt.timedelta(days=day)
	end_time = start_time + dt.timedelta(days=1)


	forecast = csep.load_catalog_forecast(
	path_to_forecasts,
	start_time=start_time, end_time=end_time,
	region=space_magnitude_region,
	filter_spatial = True,
	apply_filters=True
	)

	forecast.filters = [f'origin_time >= {forecast.start_epoch}', f'origin_time < {forecast.end_epoch}', f'magnitude >= {forecast.min_magnitude}']
	_ = forecast.get_expected_rates(verbose=True, )


	############# define observed catalog
	cat = csep.load_catalog(dic['path_to_cat'])		
	cat.name = dic['name']

	cat = cat.filter_spatial(forecast.region)
	cat = cat.filter(f'magnitude >= {min_mw}')
	cat = cat.filter(forecast.filters)	

	fn_result = work_path+'/tests_CSEP_day_'+str(day)+'_'	


	######################### TESTS
	#### Number Test
	number_test_result = csep.core.catalog_evaluations.number_test(forecast, cat)
	result_json = json.dumps(number_test_result.to_dict())
	with open(fn_result + "number.json", "w") as f:
		f.write(result_json)

	#### Spatial Test
	if cat.event_count>0:
		spatial_test_result = csep.core.catalog_evaluations.spatial_test(forecast, cat)
		result_json = json.dumps(spatial_test_result.to_dict())
		with open(fn_result + "spatial.json", "w") as f:
			f.write(result_json)

	#### Magnitude Test
	if cat.event_count>0:
		magnitude_test_result = csep.core.catalog_evaluations.magnitude_test(forecast, cat)
		result_json = json.dumps(magnitude_test_result.to_dict())
		with open(fn_result + "magnitude.json", "w") as f:
			f.write(result_json)
