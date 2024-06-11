import pandas as pd
import numpy as np
import urllib.request
import requests
import tarfile
import shutil
import re
import os
import zipfile
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import math
from scipy.ndimage import uniform_filter1d
import geopandas as gpd


def download_USGS_data(start_time, end_time, max_lat, min_lat, max_lon, min_lon, minimum_magnitude):
	url = (
		"https://earthquake.usgs.gov/fdsnws/event/1/query.csv?"
		f"starttime={start_time.strftime('%Y-%m-%dT%H:%M:%S')}&endtime={end_time.strftime('%Y-%m-%dT%H:%M:%S')}"
		f"&maxlatitude={max_lat}&minlatitude={min_lat}&maxlongitude={max_lon}"
		f"&minlongitude={min_lon}&minmagnitude={minimum_magnitude}&eventtype=earthquake&orderby=time-asc"
	)

	filename = f"raw/{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}.csv"
	os.makedirs("raw", exist_ok=True)
	urllib.request.urlretrieve(url, filename)
	return filename

def download_USGS_in_windows(start_year, start_month, end_year, end_month, window_size_months, max_lat, min_lat, max_lon, min_lon, minimum_magnitude):
	filenames = []
	current_start = datetime(start_year, start_month, 1)
	end_time = datetime(end_year, end_month, 1) + relativedelta(months=1) - timedelta(seconds=1)
	
	while current_start <= end_time:
		current_end = current_start + relativedelta(months=window_size_months) - timedelta(seconds=1)
		if current_end > end_time:
			current_end = end_time
		filenames.append(download_USGS_data(current_start, current_end, max_lat, min_lat, max_lon, min_lon, minimum_magnitude))
		current_start = current_end + timedelta(seconds=1)
	return filenames


def combine_csv_files(filenames, destination_path, chunksize=100000):
	with open(destination_path, 'w') as wfd:
		for i, f in enumerate(filenames):
			for chunk in pd.read_csv(f, chunksize=chunksize):
				chunk.to_csv(wfd, header=(i == 0), index=False)

# Function to download the file
def download_file(url, local_filename):
	with requests.get(url, stream=True) as response:
		response.raise_for_status()  # Check if the request was successful
		with open(local_filename, 'wb') as file:
			for chunk in response.iter_content(chunk_size=8192):
				file.write(chunk)
	print(f"Downloaded {local_filename}")

def extract_tar_gz(filename, extract_path):
	if not os.path.exists(extract_path):
		os.makedirs(extract_path)
	with tarfile.open(filename, "r:gz") as tar:
		tar.extractall(path=extract_path)
	print(f"Extracted to {extract_path}")

	os.rename('SCEC_DC','raw')

def extract_zip(zip_file_path, extract_path):
	if not os.path.exists(extract_path):
		os.makedirs(extract_path)
	with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
		zip_ref.extractall(extract_path)
	print(f"Extracted to {extract_path}")

	os.remove(zip_file_path)

def download_txt_file(url,destination_path):

	if not os.path.exists(destination_path):
		os.makedirs(destination_path)

	response = requests.get(url)

	# Check if the download was successful
	if response.status_code == 200:
		# Save the content of the response to a file
		with open("tmp.txt", "wb") as file:
			file.write(response.content)
		
		# Read the downloaded file line by line
		with open("tmp.txt", "r") as file:
			lines = file.readlines()
		
		# Create an empty list to store parsed data
		data = []
		
		# Loop through each line and parse the fields using regular expression
		for line in lines:
			fields = re.split(r'\s+', line.strip())
			data.append(fields)

		# Convert the parsed data into a DataFrame
		df = pd.DataFrame(data[1:], columns=data[0])

		# Merge the datetime columns into a single datetime column
		df['time'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND']])

		# Drop the original datetime columns
		df.drop(columns=['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND'], inplace=True)

		df.rename(columns={'EVENTID': 'id', 'LATITUDE': 'latitude', 'LONGITUDE': 'longitude', 'MAGNITUDE': 'magnitude'}, inplace=True)
		# Reorder columns
		# df = df[['id', 'latitude', 'longitude', 'time', 'magnitude']]

		# Save DataFrame to CSV
		df.to_csv(destination_path+'events.csv', index=False)
		os.remove('tmp.txt')

		print("CSV file created successfully.")
	else:
		print("Failed to download the file.")

	# os.rename('SCEC_DC','raw')

def combine_ascii_files(catalog_directory):
	# Merge earthquake catalogs into a single DataFrame
	all_data = []
	for file in os.listdir(catalog_directory):
		filepath = os.path.join(catalog_directory, file)
		if file.endswith('.catalog') and os.path.isfile(filepath):
			with open(filepath, 'r') as f:
				# Read the file as ASCII
				data = f.readlines()
				# Extract column names from line 10
				column_names = data[9].strip().split()
				# Remove leading '#' if present
				column_names = [name.strip('#') for name in column_names]
				# Read the remaining lines as data
				data = data[10:]
				# Create DataFrame using column names and data
				df = pd.DataFrame([line.strip().split() for line in data], columns=column_names)
				# Merge date and time columns into a single datetime column
				df['date'] = pd.to_datetime(df['YYY/MM/DD'], format='%Y/%m/%d', errors='coerce')
				df['time'] = pd.to_timedelta(df['HH:mm:SS.ss'], errors='coerce')
				df['datetime'] = df['date'] + df['time']
				# Drop the original date and time columns
				df.drop(columns=['YYY/MM/DD', 'HH:mm:SS.ss', 'date', 'time'], inplace=True)
				# Rename columns
				df.rename(columns={'EVID': 'id', 'LAT': 'latitude', 'LON': 'longitude', 'MAG': 'magnitude','datetime': 'time'}, inplace=True)
				# Reorder columns
				# df = df[['id', 'latitude', 'longitude', 'time', 'magnitude']]
				all_data.append(df)

	if len(all_data) == 0:
		print("No earthquake catalog files found in the specified directory.")
	else:
		# Concatenate all DataFrames into a single DataFrame
		merged_data = pd.concat(all_data, ignore_index=True)

		merged_data = merged_data.dropna()

		print('final file shape: ',merged_data.shape)

		# Save merged dataset to a CSV file
		merged_data.to_csv(catalog_directory+"/SCEDC_catalog.csv", index=False)
		print("Merged dataset saved successfully as a CSV file.")


def fmd(mag, mbin):

	mag = np.array(mag)

	mi = np.arange(min(np.round(mag/mbin)*mbin), max(np.round(mag/mbin)*mbin),mbin)

	nbm = len(mi)
	cumnbmag = np.zeros(nbm)
	nbmag = np.zeros(nbm)

	for i in range(nbm):
		cumnbmag[i] = sum((mag > mi[i]-mbin/2))

	cumnbmagtmp = np.append(cumnbmag,0)
	nbmag = abs(np.ediff1d(cumnbmagtmp))

	res = {'m':mi, 'cum':cumnbmag, 'noncum':nbmag}

	return res


def maxc(mag, mbin):

	FMD = fmd(mag, mbin)

	if len(FMD['noncum'])>0:
	# if True:

		Mc = FMD['m'][np.where(FMD['noncum']==max(FMD['noncum']))[0]][0]

	else:
		Mc = None

	return Mc


def azimuthal_equidistant_projection(latitude, longitude, center_latitude, center_longitude):
	R = 6371  # Earth's radius in kilometers
	phi1 = np.radians(center_latitude)
	phi2 = np.radians(latitude)
	delta_lambda = np.radians(longitude - center_longitude)

	delta_sigma = np.arccos(np.sin(phi1) * np.sin(phi2) + np.cos(phi1) * np.cos(phi2) * np.cos(delta_lambda))
	azimuth = np.arctan2(np.sin(delta_lambda), np.cos(phi1) * np.tan(phi2) - np.sin(phi1) * np.cos(delta_lambda))

	x = R * delta_sigma * np.cos(azimuth)
	y = R * delta_sigma * np.sin(azimuth)
	return x, y