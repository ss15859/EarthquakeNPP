#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# simulation of catalog continuation (for forecasting)
#
# as described by Mizrahi et al., 2021
# Leila Mizrahi, Shyam Nandan, Stefan Wiemer;
# Embracing Data Incompleteness for Better Earthquake Forecasting.
# Journal of Geophysical Research: Solid Earth.
# doi: https://doi.org/10.1029/2021JB022379
###############################################################################


import json
import logging, sys

from etas import set_up_logger
sys.path.append('etas/')
from inversion import ETASParameterCalculation
from simulation import ETASSimulation
from datetime import datetime, timedelta

set_up_logger(level=logging.INFO)

if __name__ == '__main__':
    # read configuration in
    # '../config/[dataset].json'
    # this should contain the path to the parameters_*.json file
    # that is produced when running invert_etas.py.

    # read script arguments
    with open('config/'+sys.argv[1]+'.json', 'r') as f:
        config = json.load(f)

    days_from_start = int(sys.argv[2])
    
    fn_inversion_output = config['data_path']+'/parameters_0.json'
    fn_store_simulation = sconfig['data_path']+'/day_'+days_from_start+'.csv'
    forecast_duration = 1

    # load output from inversion
    with open(fn_inversion_output, 'r') as f:
        inversion_output = json.load(f)
    
    inversion_output['three_dim']=False
    inversion_output['space_unit_in_meters']=False

    # Set day of forecast
    inversion_output['timewindow_end'] = (datetime.strptime(inversion_output['timewindow_end'], '%Y-%m-%d %H:%M:%S')+timedelta(days=days_from_start)).strftime('%Y-%m-%d %H:%M:%S')
    

    etas_inversion_reload = ETASParameterCalculation.load_calculation(
        inversion_output)

    # initialize simulation
    simulation = ETASSimulation(etas_inversion_reload)
    simulation.prepare()

    # simulate and store one catalog
    simulation.simulate_to_csv(fn_store_simulation, forecast_duration, n_simulations = 10000,i_start=0)
