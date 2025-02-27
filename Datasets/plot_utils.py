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
	"""
	Downloads earthquake data from the USGS website for a specified time range and geographical area.

	Parameters:
	- start_time (datetime): Start time for the data query.
	- end_time (datetime): End time for the data query.
	- max_lat (float): Maximum latitude of the area.
	- min_lat (float): Minimum latitude of the area.
	- max_lon (float): Maximum longitude of the area.
	- min_lon (float): Minimum longitude of the area.
	- minimum_magnitude (float): Minimum magnitude of earthquakes to include.

	Returns:
	- str: Filename of the downloaded CSV file.
	"""
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
	"""
	Downloads earthquake data in monthly windows from the USGS website.

	Parameters:
	- start_year (int): Start year for the data query.
	- start_month (int): Start month for the data query.
	- end_year (int): End year for the data query.
	- end_month (int): End month for the data query.
	- window_size_months (int): Size of each window in months.
	- max_lat (float): Maximum latitude of the area.
	- min_lat (float): Minimum latitude of the area.
	- max_lon (float): Maximum longitude of the area.
	- min_lon (float): Minimum longitude of the area.
	- minimum_magnitude (float): Minimum magnitude of earthquakes to include.

	Returns:
	- list: List of filenames of the downloaded CSV files.
	"""
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
	"""
	Combines multiple CSV files into a single CSV file.

	Parameters:
	- filenames (list): List of CSV filenames to combine.
	- destination_path (str): Path to save the combined CSV file.
	- chunksize (int): Number of rows per chunk to read at a time.

	Returns:
	- None
	"""
	with open(destination_path, 'w') as wfd:
		for i, f in enumerate(filenames):
			for chunk in pd.read_csv(f, chunksize=chunksize):
				chunk.to_csv(wfd, header=(i == 0), index=False)

# Function to download the file
def download_file(url, local_filename):
	"""
	Downloads a file from a given URL.

	Parameters:
	- url (str): URL of the file to download.
	- local_filename (str): Local path to save the downloaded file.

	Returns:
	- None
	"""
	with requests.get(url, stream=True) as response:
		response.raise_for_status()  # Check if the request was successful
		with open(local_filename, 'wb') as file:
			for chunk in response.iter_content(chunk_size=8192):
				file.write(chunk)
	print(f"Downloaded {local_filename}")

def extract_tar_gz(filename, extract_path):
	"""
	Extracts a .tar.gz file to a specified directory.

	Parameters:
	- filename (str): Path to the .tar.gz file.
	- extract_path (str): Directory to extract the contents to.

	Returns:
	- None
	"""
	if not os.path.exists(extract_path):
		os.makedirs(extract_path)
	with tarfile.open(filename, "r:gz") as tar:
		tar.extractall(path=extract_path)
	print(f"Extracted to {extract_path}")

	os.rename('SCEC_DC','raw')

def extract_zip(zip_file_path, extract_path):
	"""
	Extracts a .zip file to a specified directory.

	Parameters:
	- zip_file_path (str): Path to the .zip file.
	- extract_path (str): Directory to extract the contents to.

	Returns:
	- None
	"""
	if not os.path.exists(extract_path):
		os.makedirs(extract_path)
	with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
		zip_ref.extractall(extract_path)
	print(f"Extracted to {extract_path}")

	os.remove(zip_file_path)

def download_txt_file(url,destination_path):
	"""
	Downloads a text file from a given URL and converts it to a CSV file.

	Parameters:
	- url (str): URL of the text file to download.
	- destination_path (str): Directory to save the converted CSV file.

	Returns:
	- None
	"""
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
	"""
	Combines multiple ASCII earthquake catalog files into a single CSV file.

	Parameters:
	- catalog_directory (str): Directory containing the catalog files.

	Returns:
	- None
	"""
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
	"""
	Computes the frequency-magnitude distribution of earthquakes.

	Parameters:
	- mag (array-like): Array of earthquake magnitudes.
	- mbin (float): Magnitude bin size.

	Returns:
	- dict: Dictionary containing magnitude bins, cumulative counts, and non-cumulative counts.
	"""
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
	"""
	Determines the magnitude of completeness (Mc) from the frequency-magnitude distribution.

	Parameters:
	- mag (array-like): Array of earthquake magnitudes.
	- mbin (float): Magnitude bin size.

	Returns:
	- float: Magnitude of completeness (Mc).
	"""
	FMD = fmd(mag, mbin)

	if len(FMD['noncum'])>0:
	# if True:

		Mc = FMD['m'][np.where(FMD['noncum']==max(FMD['noncum']))[0]][0]

	else:
		Mc = None

	return Mc

def deg2rad(deg):
    return deg * np.pi / 180

def rad2deg(rad):
    return rad * 180 / np.pi

def azimuthal_equidistant_projection(lat, lon, lat0, lon0, R=6371):
	"""
	Forward azimuthal equidistant projection.
	Converts (lat, lon) to (x, y) with respect to a center (lat0, lon0).

	Parameters:
	- lat (float): Latitude to project (degrees).
	- lon (float): Longitude to project (degrees).
	- lat0 (float): Center latitude of the projection (degrees).
	- lon0 (float): Center longitude of the projection (degrees).
	- R (float): Radius of the sphere (default: Earth radius in km).

	Returns:
	- tuple: Projected coordinates (x, y) in km.
	"""
	lat, lon, lat0, lon0 = map(deg2rad, [lat, lon, lat0, lon0])
	
	delta_lambda = lon - lon0
	
	c = np.arccos(np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(delta_lambda))
	
	k = np.where(c == 0, 0, R * c / np.sin(c))
	y = k * np.cos(lat) * np.sin(delta_lambda)
	x = k * (np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(delta_lambda))
	
	return x, y