import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import argparse
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import cartopy

import csep
import csep
from csep.core import regions
from csep.utils import datasets, time_utils

from run_pycsep_tests import get_dataset_config

colors = ["#5F0F40", "#C9DAEA", "#84A07C", "#E36414", "#39A9DB", "#0081A7", "#284B63", "#FFD449"]

def plot_number_forecasts(model_paths, start_day, end_day, dataset, min_magnitude, region, test_nll_start):


	#convert start_day and end_day to datetime objects
	test_nll_start_dt = datetime.strptime(test_nll_start, '%Y-%m-%d %H:%M:%S')

	start_time_dt = test_nll_start_dt + timedelta(days=start_day)
	end_time_dt = test_nll_start_dt + timedelta(days=end_day+1)

	start_time = csep.utils.time_utils.datetime_to_utc_epoch(start_time_dt)
	end_time = csep.utils.time_utils.datetime_to_utc_epoch(end_time_dt)


	# Create space-magnitude region
	min_mw = min_magnitude
	max_mw = 7.65
	dmw = 0.1
	magnitudes = regions.magnitude_bins(min_mw, max_mw, dmw)
	space_magnitude_region = regions.create_space_magnitude_region(region, magnitudes)


	# Define observed catalog
	cat = csep.load_catalog(catalog_path)
	cat = cat.filter(f'magnitude >= {min_mw}')
	cat = cat.filter(f'origin_time >= {start_time}')
	cat = cat.filter(f'origin_time < {end_time}')


	catalog = cat.to_dataframe(with_datetime=True)

	catalog['datetime'] = catalog['datetime'].dt.tz_localize(None)

	# Compute cumulative count
	catalog['cumulative_count'] = range(1, len(catalog) + 1)

	#make a datapoint at the end of the last day with no increase in cumulative count
	catalog = catalog._append({'datetime': end_time_dt, 'cumulative_count': catalog['cumulative_count'].iloc[-1]}, ignore_index=True)

	# Set up plot
	fig, ax1 = plt.subplots(figsize=(10, 8))
	ax1.set_facecolor((0.95, 0.95, 0.95))

	# Create twin axis for magnitudes
	ax2 = ax1.twinx()
	z = (4**catalog['magnitude'])*0.05
	ax1.scatter(catalog['datetime'], catalog['magnitude'], color=colors[3], s=z, alpha=0.8, zorder=-10)

	ax2.plot(catalog['datetime'], catalog['cumulative_count'], label='Cumulative Count', color=colors[0], lw=2, zorder=2)

	# Move ax1 y-axis to the right
	ax1.yaxis.set_label_position("right")
	ax1.yaxis.tick_right()

	# Move ax2 y-axis to the left
	ax2.yaxis.set_label_position("left")
	ax2.yaxis.tick_left()

	# Process each model
	for i, model_path in enumerate(model_paths):
		model = model_path['model']
		forecast_folder = model_path['test_results_path']
		forecast_quantiles = []


		for day in range(start_day, end_day+1):
			date = test_nll_start_dt + timedelta(days=day)
			forecast_file = os.path.join(forecast_folder, f'tests_CSEP_day_{day}_number.json')
			
			if os.path.exists(forecast_file):
				with open(forecast_file, 'r') as f:
					forecast_data = json.load(f)
					forecast_data = forecast_data['test_distribution']
				
				# Compute 95% quantile
				lower_quantile = np.percentile(forecast_data, 2.5)  # Lower bound of 95% interval
				upper_quantile = np.percentile(forecast_data, 97.5)  # Upper bound of 95% interval
				forecast_quantiles.append((date, lower_quantile, upper_quantile))
				
			else:
				print(f"File {forecast_file} not found")
				forecast_quantiles.append((date, None, None))  # Handle missing files

		

		# Prepare cumulative count data for shifting vlines
		# extract only the date from the datatime
		catalog['date'] = catalog['datetime'].dt.date
		daily_cumulative = catalog.groupby('date')['cumulative_count'].max().shift(1, fill_value=0)  # Previous day's cumulative count

		# Merge forecast data with cumulative count offsets
		forecast_df = pd.DataFrame(forecast_quantiles, columns=['date', 'lower', 'upper'])
		forecast_df['cumulative_offset'] = forecast_df['date'].map(daily_cumulative)

		# Plot adjusted vertical lines and color them based on model, offset so models don't overlap
		valid_forecast_df = forecast_df.dropna()  # Remove rows with missing data
		display_legend = True
		for _, row in valid_forecast_df.iterrows():
			cumulative_offset = row['cumulative_offset']  # Get the cumulative offset for the day
			ax2.vlines(
				x=row['date'] + pd.Timedelta(days=1 + i*0.19),  # Shift to the end of the day
				ymin=row['lower'] + cumulative_offset, 
				ymax=row['upper'] + cumulative_offset, 
				color = 'red' if model == 'SMASH' else 'green' if model == 'ETAS' else 'yellow',
				alpha=1, 
				lw=5, 
				label=f'{model} 95% Forecast Number Distribution' if display_legend else None 
			)
			display_legend = False  # Only add label once per model


	ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
	# Add labels and legend
	ax2.set_xlabel('Date', fontsize=12)
	ax2.set_ylabel('Cumulative Count',color=colors[0], fontsize=12)
	ax2.tick_params(axis='y', labelsize=12, colors=colors[0])
	ax1.set_ylabel('Magnitude',color=colors[3], fontsize=12)
	ax1.tick_params(axis='y', labelsize=12, colors=colors[3])
	ax2.legend(fontsize=10, loc='lower right')
	plt.tight_layout()
	# save the plot with the dataset name, models used, and the start and end day
	plt.savefig(f'plots/number_{dataset}_{"_".join([model["model"] for model in model_paths])}_{start_day}_{end_day}.png')



def plot_spatial_forecasts(model_paths, start_day, end_day, dataset, min_magnitude, region, test_nll_start):
	# Convert start_day and end_day to datetime objects
	test_nll_start_dt = datetime.strptime(test_nll_start, '%Y-%m-%d %H:%M:%S')

	start_time_dt = test_nll_start_dt + timedelta(days=start_day)
	end_time_dt = test_nll_start_dt + timedelta(days=end_day + 1)

	start_time = csep.utils.time_utils.datetime_to_utc_epoch(start_time_dt)
	end_time = csep.utils.time_utils.datetime_to_utc_epoch(end_time_dt)

	# Create space-magnitude region
	min_mw = min_magnitude
	max_mw = 7.65
	dmw = 0.1
	magnitudes = regions.magnitude_bins(min_mw, max_mw, dmw)
	space_magnitude_region = regions.create_space_magnitude_region(region, magnitudes)

	# Define observed catalog
	cat = csep.load_catalog(model_paths[0]['catalog_path'])
	cat = cat.filter(f'magnitude >= {min_mw}')
	cat = cat.filter(f'origin_time >= {start_time}')
	cat = cat.filter(f'origin_time < {end_time}')

	# Create a figure with subplots for each model
	num_models = len(model_paths)
	fig, axes = plt.subplots(1, num_models, figsize=(16, 8), subplot_kw={'projection': cartopy.crs.Mercator()})
	if num_models == 1:
		axes = [axes]  # Ensure axes is always a list, even for a single model
	
	plt.subplots_adjust(wspace=0.6) 

	for ax, model_path in zip(axes, model_paths):
		model = model_path['model']
		

		catalog_forecast_df = pd.DataFrame()

		for day in range(start_day, end_day + 1):
			date = test_nll_start_dt + timedelta(days=day)
			

			try:
				path_to_forecast_day = os.path.join(model_path['forecast_dir'], f'CSEP_day_{day}.csv')
				df = pd.read_csv(path_to_forecast_day)
				catalog_forecast_df = pd.concat([catalog_forecast_df, df], ignore_index=True)
			except FileNotFoundError:
				print(f'Could not find forecast for model: {model}, day: {day}')
				continue

		catalog_forecast_df = catalog_forecast_df.sort_values(by=['catalog_id', 'time_string'])
		catalog_forecast_df.to_csv('tmp.csv', index=False)

		start_time = test_nll_start_dt + timedelta(days=start_day)
		end_time = start_time + timedelta(days=1)

		forecast = csep.load_catalog_forecast(
			'tmp.csv',
			start_time=start_time, end_time=end_time,
			region=space_magnitude_region,
			filter_spatial=True,
			apply_filters=True
		)
		forecast.filters = [f'origin_time >= {forecast.start_epoch}', f'origin_time < {forecast.end_epoch}', f'magnitude >= {forecast.min_magnitude}']
		expected_rates = forecast.get_expected_rates(verbose=True)

		# Plot expected rates on the current axis
		args_forecast = {'title': f'{model} Spatial Forecast',
						 'grid_labels': True,
						 'basemap': 'ESRI_imagery',
						 'cmap': 'rainbow',
						 'alpha_exp': 0.5,
						 'projection': cartopy.crs.Mercator(),
						 'clim': [-3.5, 0]}
		expected_rates.plot(ax=ax, plot_args=args_forecast)
		args_catalog = {'basemap': 'ESRI_terrain',
				  	'title': f'{model}',
					'markercolor': 'grey',
					'markersize': 1,
					'title_size': 10,
					'grid':True,
					'grid_labels': True,
					'grid_fontsize': 10}
		cat.plot(ax=ax, plot_args=args_catalog)

	# Adjust layout and save the combined plot
	fig.suptitle(f'{dataset} {start_time.strftime("%d/%m/%y")} - {end_time.strftime("%d/%m/%y")}', fontsize=16)
	plt.savefig(f'plots/spatial_combined_{dataset}_{start_day}_{end_day}.png')
	# plt.show()

def retrieve_test_results(test_type, model_paths, start_day, end_day, dataset, min_magnitude, region, test_nll_start):

	test_nll_start_dt = datetime.strptime(test_nll_start, '%Y-%m-%d %H:%M:%S')

	start_time_dt = test_nll_start_dt + timedelta(days=start_day)
	end_time_dt = test_nll_start_dt + timedelta(days=end_day+1)

	start_time = csep.utils.time_utils.datetime_to_utc_epoch(start_time_dt)
	end_time = csep.utils.time_utils.datetime_to_utc_epoch(end_time_dt)


	# Create space-magnitude region
	min_mw = min_magnitude
	max_mw = 7.65
	dmw = 0.1
	magnitudes = regions.magnitude_bins(min_mw, max_mw, dmw)
	space_magnitude_region = regions.create_space_magnitude_region(region, magnitudes)


	# Define observed catalog
	cat = csep.load_catalog(catalog_path)
	cat = cat.filter(f'magnitude >= {min_mw}')
	cat = cat.filter(f'origin_time >= {start_time}')
	cat = cat.filter(f'origin_time < {end_time}')


	catalog = cat.to_dataframe(with_datetime=True)

	catalog['datetime'] = catalog['datetime'].dt.tz_localize(None)

	for i, model_path in enumerate(model_paths):
		model = model_path['model']
		forecast_folder = model_path['test_results_path']
		bind_res_df = pd.DataFrame()

		for day in range(start_day, end_day+1):
			date = test_nll_start_dt + timedelta(days=day)
			forecast_file = os.path.join(forecast_folder, f'tests_CSEP_day_{day}_{test_type}.json')
			
			try:
				with open(forecast_file, 'r') as f:
					forecast_data = json.load(f)
				
				# Compute 95% quantile
				lower_quantile = np.percentile(forecast_data['test_distribution'], 2.5)  # Lower bound of 95% interval
				upper_quantile = np.percentile(forecast_data['test_distribution'], 97.5)  # Upper bound of 95% interval
				report_string  = forecast_data["obs_catalog_repr"]
				# extract the event count from the report string by finding the number following "Event Count: " and preceding "\n"
				n_test_obs = float(report_string[report_string.find("Event Count: ") + len("Event Count: "): ])
				bind_res_df = bind_res_df._append({'date': date, 'obs': forecast_data['observed_statistic'], 'low_ci': lower_quantile, 'upper_ci': upper_quantile, 'n_test_obs': n_test_obs, 'gamma': forecast_data['quantile'][0]}, ignore_index=True)


			except:
				print(f"File {forecast_file} not found")
				bind_res_df = bind_res_df._append({'date': date, 'obs': None, 'low_ci': None, 'upper_ci': None, 'n_test_obs': None, 'gamma': None}, ignore_index=True)  # Handle missing files

	return bind_res_df, catalog
	

def plot_gamma_m_res(bind_res_df, cat_df, test_type):

	bind_res_df2 = bind_res_df[ (bind_res_df['n_test_obs'].notnull())]
	# make sure n_test_obs are floats and log the value
	bind_res_df2['n_test_obs'] = bind_res_df2['n_test_obs'].astype(float)

	colors = ['red' if ( bind_res_df2['obs'].iloc[idx] <= bind_res_df2['low_ci'].iloc[idx] or bind_res_df2['obs'].iloc[idx] >= bind_res_df2['upper_ci'].iloc[idx]) else 'green' for idx in range(len(bind_res_df2['date']))]

	fig, axs = plt.subplot_mosaic([['m_test', '.'],
								
								['scatter', 'histy'],
								['scatter_events', '.']
								],
								figsize=(10, 5),
								width_ratios=(4, 1), height_ratios=(1, 4, 1),
								layout='constrained')
	ax = axs['scatter']
		# the scatter plot:
	ax.scatter(bind_res_df2['date'], bind_res_df2['gamma'], s = bind_res_df2['n_test_obs'], c = colors, edgecolors="black", linewidth=0.2, zorder = 10)
	ax.set_ylabel('$\gamma_m$')
	ax.axhline(y = 0.025, c = 'grey', alpha = 0.5, linestyle = '--')
	ax.axhline(y = 0.975, c = 'grey', alpha = 0.5, linestyle = '--')
	ax.annotate('(b)', xy=(0.02, 0.9), xycoords='axes fraction', ha='center', va='center', fontsize=10)
	
	#ax.set_xticks([])
	#for dd in large_mag['date']:
	#    ax.axvline(x = dd, alpha = 0.2, c = 'red', zorder = 0)

	ax_histy =  axs['histy']
	ax_histy.tick_params(axis="y", labelleft=False)
		# now determine nice limits by hand:

	ax_histy.hist(bind_res_df2['gamma'], bins=30, orientation='horizontal', color="gray", edgecolor="black")
	ax_histy.set_xlabel('Frequency')
	ax_histy.annotate('(d)', xy=(0.2, 0.89), xycoords='axes fraction', ha='center',
					va='center', fontsize=10,
					bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
	annot_text = 'Red $\%$ = ' + str(np.round(np.mean([cc == 'red' for cc in colors]), 3))
	ax_histy.annotate(annot_text, xy=(0.7, 0.1), xycoords='axes fraction', ha='center', va='center', fontsize=10,
			bbox=dict(facecolor="lightblue", edgecolor="red", alpha= 1))

	ax_scatter_ev = axs['scatter_events']
	ax_scatter_ev.tick_params(axis="x", labelbottom=False)
	ax_scatter_ev.scatter(cat_df['datetime'], cat_df['magnitude'], s = np.exp(cat_df['magnitude'])/20, c = 'red', alpha = 0.5, edgecolors="black", linewidth=0.5)
	ax_scatter_ev.vlines(x = cat_df['datetime'], ymin = cat_df['magnitude'].min(),
			ymax = cat_df['magnitude'], zorder = 0, linewidth = 0.5, alpha = 0.2, color = 'black')
	ax_scatter_ev.set_ylabel('Magnitude')
	ax_scatter_ev.annotate('(c)', xy=(0.02, 0.8), xycoords='axes fraction', ha='center', va='center', fontsize=10)

	ax_n_test = axs['m_test']
	ax_n_test.tick_params(axis="x", labelbottom=False)
	ax_n_test.vlines(x = bind_res_df2['date'], ymin = bind_res_df2['low_ci'],
			ymax = bind_res_df2['upper_ci'] , color = 'grey', zorder = 0, linewidth = 0.5)
	ax_n_test.scatter(bind_res_df2['date'], bind_res_df2['obs'], c = colors, zorder = 10, s = bind_res_df2['n_test_obs']/4, edgecolors="black", linewidth=0.2)
	ax_n_test.annotate('(a)', xy=(0.02, 0.8), xycoords='axes fraction', ha='center', va='center', fontsize=10)
	ax_n_test.set_ylabel('$L_m$')
	if test_type == 'number':
		ax_n_test.semilogy()


	# plt.savefig('test.png', dpi = 500)
	plt.show()



	

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Plot forecast number distribution for a specified day range")
	parser.add_argument('--models', type=str, nargs='+', required=True, 
						choices=['SMASH', 'ETAS', 'DSTPP'], 
						help='Model names (SMASH, ETAS, DSTPP)')
	parser.add_argument('--dataset', type=str, required=True, 
						choices=['ComCat', 'WHITE', 'SCEDC', 'SanJac', 'SaltonSea'],
						help='Dataset name')
	parser.add_argument('--start_day', type=int, required=True, help="Start day of the range")
	parser.add_argument('--end_day', type=int, required=True, help="End day of the range")
	parser.add_argument('--test_type', type=str, default="number", choices=['number', 'spatial', 'pseudolikelihood', 'magnitude'], help="Type of test to plot")
	parser.add_argument('--HPC', action='store_true', help="Run on HPC")
	args = parser.parse_args()

	Mc, region, test_nll_start = get_dataset_config(args.dataset)
	
	# Format Mc value
	Mc_str = f"{int(Mc * 10):02d}"
	
	# Set default paths based on model and dataset if not provided
	if args.HPC:
		forecast_path = '/user/work/ss15859/'
	else:
		forecast_path = '../'

	model_paths = []
	for model in args.models:
		if model == 'SMASH':
			forecast_dir = forecast_path + f'/SMASH/daily_forecasts/{args.dataset}'
			plots_path = forecast_path + f'/SMASH/daily_forecasts/{args.dataset}/plots'
			test_results_path = forecast_path + f'/SMASH/daily_forecasts/{args.dataset}/test_results'
		elif model == 'ETAS':
			forecast_dir = forecast_path + f'/ETAS/output_data_{args.dataset}_{Mc_str}/forecasts'
			plots_path = forecast_path + f'/ETAS/output_data_{args.dataset}_{Mc_str}/plots'
			test_results_path = forecast_path + f'/ETAS/output_data_{args.dataset}_{Mc_str}/test_results'
		elif model == 'DSTPP':
			forecast_dir = forecast_path + f'Spatio-temporal-Diffusion-Point-Processes/daily_forecasts/{args.dataset}'
			plots_path = forecast_path + f'Spatio-temporal-Diffusion-Point-Processes/daily_forecasts/{args.dataset}/plots'
			test_results_path = forecast_path + f'Spatio-temporal-Diffusion-Point-Processes/daily_forecasts/{args.dataset}/test_results'

		catalog_path = forecast_path + f'/ETAS/output_data_{args.dataset}_{Mc_str}/CSEP_format_catalog.csv'
		model_paths.append({
			'model': model,
			'forecast_dir': forecast_dir,
			'catalog_path': catalog_path,
			'plots_path': plots_path,
			'test_results_path': test_results_path
		})

	# plot_number_forecasts(
	# 	model_paths=model_paths,
	# 	start_day=args.start_day,
	# 	end_day=args.end_day,
	# 	dataset=args.dataset,
	# 	min_magnitude=Mc,
	# 	region=region,
	# 	test_nll_start=test_nll_start
	# )


	# plot_spatial_forecasts(
	# 	model_paths=model_paths,
	# 	start_day=args.start_day,
	# 	end_day=args.end_day,
	# 	dataset=args.dataset,
	# 	min_magnitude=Mc,
	# 	region=region,
	# 	test_nll_start=test_nll_start
	# )

	
	bind_res_df, cat_df = retrieve_test_results(args.test_type, model_paths, args.start_day, args.end_day, args.dataset, Mc, region, test_nll_start)
	plot_gamma_m_res(bind_res_df, cat_df, args.test_type)