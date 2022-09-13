"""
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from spotmarkets import EmarketModel, time_operator_factory


import dill as pickle 
from helpfuncs import config

import time
import sys
from numpy import genfromtxt

import copy

# Initialize demand shock 
if __name__ == '__main__':

	# load array of parameter values for each model 
	settings_file = "baseline_3"
	array = genfromtxt('Settings/ERCOT/{}.csv'\
			.format(settings_file), delimiter=',')[1:]

	

	model_name = 'ERCOT'
	demand_name = 'errors_demand'
	
	U = pickle.load(open("Settings/{}/seed_u.pkl"\
					.format(model_name),"rb"))

	demand_shocks = np.genfromtxt('Settings/{}.csv'.format(demand_name), delimiter=',')


	parameters = array[1]
	print(parameters)
	og = EmarketModel(s_supply = parameters[1], #variance deviation of supply 
					mu_supply = parameters[0],
					grid_size = 20, #grid size of storage grid
					grid_max_x = 100, #initial max storge (redundant)
					D_bar = parameters[6], #demand parameter D_bar
					r_s = parameters[2]*1E9,#cost str cap (USD/GwH)
					r_k = parameters[3]*1E9, #cost gen ap(USD/Gw)
					grid_size_s  = 2, #number of supply shocks
					grid_size_d  = 2, #number of demand shocks
					zeta_storage = parameters[4], # pipe constraint
					eta_demand   = parameters[5], #demand elas.
					s_demand = parameters[8], 
					U = U, demand_shocks = demand_shocks)


	GF_RMSE, GF_star, TC_star, G, F  = time_operator_factory(og) 


	tol_TC = 1e-2
	#tol_brent = 1e-4 #second stage tol of brent's method to find S_bar
	#tol_brent_1 = .5 #first stage tol of brent's method to find S_bar
	# number of digits to round the delta storage to determine whether it bind
	tol_pi = 4


	K = 100
	S = 100
	bounds = ((0.1, 1000), (0.1, 1000))

	#error,rho_star = GF_RMSE(K,S,tol_TC,tol_pi)

	def error_mkt(cap):
		print('eval_func')
		error,rho_star = GF_RMSE(cap[0],cap[1],tol_TC,tol_pi)
		return error

	#import tensorflow as tf
	#a1 = np.array([K, S])

	#start = time.time()
	#error_mkt_tf = tf.numpy_function(error_mkt, [a1], Tout=tf.float64)

	#print('donetf')
	#print(time.time() - start)

	#start = time.time()
	#error_normal = GF_RMSE(a1[0],a1[1],tol_TC,tol_pi)
	#print(time.time() - start)

	#error_mkt_tf(a1).numpy()

	import scipy

	res = scipy.optimize.differential_evolution(error_mkt, bounds = bounds, workers = 4)



