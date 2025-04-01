from run_pycsep_tests import get_dataset_config
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import sys


def plot_int_lambd(model_paths, dataset, start_day, end_day, test_nll_start):
	"""
	Plot the log scores for the specified models and dataset
	"""
	
	#loop round the models and plot the log scores
	for model_path in model_paths:
		model = model_path['model']
		results_file = model_path['results_file']
		#read the results file
		df = pd.read_csv(results_file)
		df['time'] = pd.to_datetime(df['time'])
		df = df.sort_values(by='time')
		df = df[df['magnitude'] >= Mc]

		test_nll_start = pd.to_datetime(test_nll_start)

		if start_day is not None and end_day is not None:
			df = df[(df['time'] >= test_nll_start + pd.Timedelta(days=start_day)) & 
					(df['time'] <= test_nll_start + pd.Timedelta(days=end_day))]
		else:
			df = df[df['time'] >= test_nll_start]


		# scatterplot the cum sum of column "int_lambd" against the datetime column
		plt.step(df['time'], df['int_lambd'].cumsum(), label=model, linewidth=2)
		plt.xlabel('Time')
		plt.ylabel('Cumulative Events')

	plt.step(df['time'], range(1,len(df)+1),color='black', label='ComCat', linewidth=2)
	plt.legend()
	if start_day is not None and end_day is not None:
		plt.savefig(f'plots/int_lambd_{dataset}_{start_day}_{end_day}.pdf')
	else:
		plt.savefig(f'plots/int_lambd_{dataset}.pdf')
	plt.close()


def plot_lambd_star(model_paths, dataset, start_day, end_day, test_nll_start):
	"""
	Plot the log scores for the specified models and dataset
	"""
	
	#loop round the models and plot the log scores
	for model_path in model_paths:
		model = model_path['model']
		results_file = model_path['results_file']
		#read the results file
		df = pd.read_csv(results_file)
		df['time'] = pd.to_datetime(df['time'])
		df = df.sort_values(by='time')
		df = df[df['magnitude'] >= Mc]

		test_nll_start = pd.to_datetime(test_nll_start)

		if start_day is not None and end_day is not None:
			df = df[(df['time'] >= test_nll_start + pd.Timedelta(days=start_day)) & 
					(df['time'] <= test_nll_start + pd.Timedelta(days=end_day))]
		else:
			df = df[df['time'] >= test_nll_start]

		# scatterplot the cum sum of column "int_lambd" against the datetime column
		plt.step(df['time'], df['lambd_star'], label=model, linewidth=2)
		plt.xlabel('Time')
		plt.ylabel('Intensity')

	
	plt.legend()
	if start_day is not None and end_day is not None:
		plt.savefig(f'plots/lambd_star{dataset}_{start_day}_{end_day}.pdf')
	else:
		plt.savefig(f'plots/lambd_star{dataset}.pdf')
	plt.close()


def plot_CIG(model_paths, dataset, start_day, end_day, test_nll_start):


	for model_path in model_paths:
		model = model_path['model']
		results_file = model_path['results_file']
		#read the results file
		df = pd.read_csv(results_file)
		df['time'] = pd.to_datetime(df['time'])
		df = df.sort_values(by='time')
		df = df[df['magnitude'] >= Mc]

		test_nll_start = pd.to_datetime(test_nll_start)

		if start_day is not None and end_day is not None:
			df = df[(df['time'] >= test_nll_start + pd.Timedelta(days=start_day)) & 
					(df['time'] <= test_nll_start + pd.Timedelta(days=end_day))]
		else:
			df = df[df['time'] >= test_nll_start]

		if model == 'ETAS':
			ETAS_df = df

		else:
			df['IG'] = df['TLL'] - ETAS_df['TLL']
			plt.step(df['time'], df['IG'].cumsum(), label=model+" - ETAS", linewidth=2)
			plt.xlabel('Time')
			plt.ylabel('Cumulative IG')

	plt.legend()
	if start_day is not None and end_day is not None:
		plt.savefig(f'plots/TCIG{dataset}_{start_day}_{end_day}.pdf')
	else:
		plt.savefig(f'plots/TCIG{dataset}.pdf')
	plt.close()


	


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Plot forecast number distribution for a specified day range")
	parser.add_argument('--models', type=str, nargs='+', required=True, 
						choices=['DeepSTPP', 'ETAS', 'AutoSTPP'], 
						help='Model names (DeepSTPP, ETAS, AutoSTPP)')
	parser.add_argument('--dataset', type=str, required=True, 
						choices=['ComCat', 'WHITE', 'SCEDC', 'SanJac', 'SaltonSea'],
						help='Dataset name')
	parser.add_argument('--start_day', type=int, required=False, help="Start day of the range")
	parser.add_argument('--end_day', type=int, required=False, help="End day of the range")
	parser.add_argument('--seed', type=int, required=False, help="seed")
	args = parser.parse_args()

	Mc, region, test_nll_start = get_dataset_config(args.dataset)
	
	# Format Mc value
	Mc_str = f"{int(Mc * 10):02d}"
	
	# Set default paths based on model and dataset if not provided
	forecast_path = '../'

	model_paths = []
	for model in args.models:
		if model == 'DeepSTPP':
			results_file = forecast_path + f'/AutoSTPP/output_data/{args.dataset}_{Mc_str}_deep_stpp_seed_{args.seed}/augmented_catalog.csv'
		elif model == 'ETAS':
			results_file = forecast_path + f'/ETAS/output_data_{args.dataset}_{Mc_str}/augmented_catalog.csv'
		elif model == 'AutoSTPP':
			results_file = forecast_path + f'/AutoSTPP/output_data/{args.dataset}_{Mc_str}_autoint_stpp_seed_{args.seed}/augmented_catalog.csv'

		model_paths.append({
			'model': model,
			'results_file': results_file
		})
		
	plot_int_lambd(model_paths, args.dataset, args.start_day, args.end_day, test_nll_start)

	plot_CIG(model_paths, args.dataset, args.start_day, args.end_day, test_nll_start)

	plot_lambd_star(model_paths, args.dataset, args.start_day, args.end_day, test_nll_start)