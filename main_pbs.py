"""
Module solves array of Spotmarket models 
Root finding for eqm. performed using Cross-Entropy method (see Kroese et al)

Script must be run using Mpi with cpus = 96 * number
of models to be solved.


Example (on Gadi normal compute node):

module load python3/3.7.4
module load openmpi/4.0.5

 
mpiexec -n 384 python3 -m mpi4py main_pbs.py baseline_3 baseline_3

"""

import numpy as np
from interpolation import interp
from quantecon.optimize.scalar_maximization import brent_max
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from itertools import product
from pathlib import Path
from numba import njit, prange
from pathos.multiprocessing import ProcessingPool

from interpolation.splines import UCGrid, CGrid, nodes, eval_linear
from spotmarkets import EmarketModel, time_operator_factory


import dill as pickle 
from helpfuncs import config
from scipy.optimize import root

import time
import sys
from numpy import genfromtxt

import copy

# Initialize demand shock 
if __name__ == '__main__':

	from mpi4py import MPI as MPI4py
	world = MPI4py.COMM_WORLD
	world_size = world.Get_size()
	world_rank = world.Get_rank()

	# load array of parameter values for each model 
	settings_file = sys.argv[1]
	array = genfromtxt('Settings/{}.csv'\
			.format(settings_file), delimiter=',')[1:]
	model_name = 'main_v_2'
	sim_name = sys.argv[2]
	N = 384
	N_elite = 9

	U = pickle.load(open("/scratch/kq62/{}/seed_u.pkl"\
					.format(model_name),"rb"))
	
	# Now make the communicator classes 
	# split len(array)*N cores across len(array) classes
	# each layer 1 communicator class solves a paramterisation of the model 

	block_size_layer_1 = int(N)
	blocks_layer_1 = world_size/block_size_layer_1
	color_layer_1 = int(world_rank/block_size_layer_1)
	key_layer1 = int(world_rank%block_size_layer_1)
	layer_1_comm = world.Split(color_layer_1,key_layer1)

	if layer_1_comm.rank == 0:
		parameters = array[int(color_layer_1)]
		print(parameters)
		og = EmarketModel(s_supply = parameters[1], #variance deviation of supply 
							mu_supply = parameters[0],
							grid_size = 100, #grid size of storage grid
							grid_max_x = 100, #initial max storge (redundant)
							D_bar = parameters[6], #demand parameter D_bar
							r_s = parameters[2]*1E9,#cost str cap (USD/GwH)
							r_k = parameters[3]*1E9, #cost gen ap(USD/Gw)
							grid_size_s  = 5, #number of supply shocks
							grid_size_d  = 5, #number of demand shocks
							zeta_storage = parameters[4], # pipe constraint
							eta_demand   = parameters[5], #demand elas.
							U = U)
	else:
		og = None 

	og = layer_1_comm.bcast(og, root =0 )

	# Import functions to generate first stage profits 
	GF_RMSE, GF_star, TC_star, G, F  = time_operator_factory(og) 
	
	# Initial grid on config file 
	#config.grid_old  = og.grid 
	#Initialize initial value of initial price guess
	#config.rho_global_old  = np.tile(np.ones(og.grid_size)*100, (og.grid_size_s*og.grid_size_d,1)) 
	#config.S_init  = 200 

	# Set tolerances 
	# tol for pricing function iteration 
	tol_TC = 1e-3 
	#tol_brent = 1e-4 #second stage tol of brent's method to find S_bar
	#tol_brent_1 = .5 #first stage tol of brent's method to find S_bar
	# number of digits to round the delta storage to determine whether it bind
	tol_pi = 4
	#tol_K = 1e-4 #tol for value of eqm. K
	# set initial stage to use first stage brent tol 
	config.toggle_GF = 0 
	tol_cp = 1e-2
	max_iter_xe = 100
	eta_b = 0.8

	# Cross entropy method

	# set bounds for draws on each cpu
	K_bounds = [1E-10,500]
	s_bounds = [1E-10,500]

	# Initialise empty variables
	i = 0
	mean_errors = 1
	mean_errors_all = 1

	# Initial uniform draw
	K = np.random.uniform(K_bounds[0], K_bounds[1])
	S = np.random.uniform(s_bounds[0], s_bounds[1])

	# Empty array to fill with next iter vals
	Kstats = np.empty(3, dtype=np.float64)
	Sstats = np.empty(3, dtype=np.float64)
	cov_matrix = np.empty((2, 2), dtype=np.float64)

	while mean_errors > tol_cp and i < max_iter_xe:
		# Evaluate mean error
		error = GF_RMSE(K,S,tol_TC,tol_pi)

		if np.isnan(error):
			error = 1e100

		# Layer one waits till all draws computed 
		layer_1_comm.Barrier()

		# Gather list of RMSE and capital stocks on layer 1 head
		indexed_errors = layer_1_comm.gather(error, root=0)
		parameter_K = layer_1_comm.gather(K, root=0)
		parameter_S = layer_1_comm.gather(S, root=0)

		if layer_1_comm.rank == 0:

			#  sort K and S vals according to errors
			#  else, append i-1 K and S vals and sort
			if i == 0:
				parameter_K_sorted = np.take(parameter_K,
											 np.argsort(indexed_errors))
				parameter_S_sorted = np.take(parameter_S,
											 np.argsort(indexed_errors))
				indexed_errors_sorted = np.sort(indexed_errors)
			else:
				indexed_errors = np.append(indexed_errors_sorted,\
														indexed_errors)

				parameter_K_sorted = np.take(np.append(parameter_K_sorted,\
														parameter_K),\
													np.argsort(indexed_errors))
				
				parameter_S_sorted = np.take(np.append(parameter_S_sorted,\
														 parameter_S),\
													np.argsort(indexed_errors))
				indexed_errors_sorted = np.sort(indexed_errors)

			# Take the elite set 
			elite_errors = indexed_errors_sorted[0: N_elite]
			elite_K = parameter_K_sorted[0: N_elite]
			elite_S = parameter_S_sorted[0: N_elite]
			elite_vec = np.stack((elite_K, elite_S))

			# Smooth of i> 0
			if i == 0:
				eta_b1 = 0
			else:
				eta_b1 = eta_b

			# Generate new covariance matrix from elite set 
			# Next period draw from smoothing with previous cov matrix
			cov_matrix_new = np.cov(elite_vec, rowvar=True)
			cov_matrix  = eta_b1*cov_matrix_new + (1-eta_b1)*cov_matrix

			# Generate the men gen capital 
			Kstats_new = np.array([np.mean(elite_K), np.std(
				elite_K), np.std(parameter_K_sorted)], dtype=np.float64)
			Kstats = eta_b1*Kstats_new + (1-eta_b1)*Kstats

			Sstats_new = np.array([np.mean(elite_S), np.std(
				elite_S), np.std(parameter_S_sorted)], dtype=np.float64)
			Sstats = eta_b1*Sstats_new + (1-eta_b1)*Sstats

			# Error in terms of max. mean difference between iteration of 
			# capital stocks 
			mean_errors_all = max(np.abs(Kstats[0]-Kstats_new[0]),\
										 np.abs(Sstats[0]-Sstats_new[0]))

			print('Rank {}, CE X-entropy iteration {}, mean gen cap {},\
					 mean stor cap {}, mean error {}'
				.format(color_layer_1, i, Kstats[0], Sstats[0], mean_errors_all))
			print('Max covariance error is {}'.format(np.max(cov_matrix)))
		else:
			pass

		# Broad cast means and covariance matrix from head of layer 1
		layer_1_comm.Barrier()
		layer_1_comm.Bcast(Kstats, root=0)
		layer_1_comm.Bcast(Sstats, root=0)
		layer_1_comm.Bcast(cov_matrix, root=0)
		mean_errors_all = layer_1_comm.bcast(mean_errors_all, root=0)

		#cov_matrix = np.diag(np.diag(cov_matrix))

		# Take new draw on each layer 1 node and clip 
		draws = np.random.multivariate_normal(np.array([Kstats[0],
														Sstats[0]]),
											  cov_matrix)
		K = min(max(K_bounds[0], draws[0]), K_bounds[1])
		S = min(max(s_bounds[0], draws[1]), s_bounds[1])
		
		# Error in terms of gen cap mean difference (mean_errors_all)
		# or covariance matrix max 
		#mean_errors = np.max(cov_matrix)
		mean_errors = copy.copy(mean_errors_all)
		i += 1


	# Once iteration complete, save results 
	if layer_1_comm.rank == 0:

		rho_star = TC_star(config.rho_global_old ,K, S, tol_TC, config.grid)
		og.K  = K
		og.rho_star = rho_star
		og.S_bar_star = S
		og.solved_flag = 1 
		og.grid = np.linspace(og.grid_min_s, S, og.grid_size)

		Path("/scratch/kq62/{}/{}/".format(model_name,settings_file))\
										.mkdir(parents=True, exist_ok=True)
		pickle.dump(og, open("/scratch/kq62/{}/{}/{}_{}_endog.mod"\
								.format(model_name, settings_file,sim_name,\
													color_layer_1),"wb"))
		

	#end             = time.time()
	#og.time_exog    = end-start 

	"""
	Run model with exogneous stock of capital. 
	Save as model_a
	"""
	
	#start           = time.time()

	#G_star          = lambda S_bar:  G(K_init, S_bar, tol_TC, tol_pi)
   # S_bar_star      = brentq(G_star, 1e-10, 2000, xtol = tol_brent)
	
	#rho_star        = TC_star(config.rho_global_old ,K_init, S_bar_star, tol_TC, config.grid)

	#og.K            = K_init
	#og.rho_star     = rho_star
	#og.S_bar_star   = S_bar_star
	#og.solved_flag  = 1 
	#og.grid         = config.grid
	#end             = time.time()
	#og.time_exog    = end-start 


	#pickle.dump(og, open("/scratch/kq62/array_6a_{}.mod".format(index),"wb"))


	"""
	Run model with endogenous stock of capital. 
	"""
	
   #config.toggle_GF =0
	#start = time.time()
	#K_star = fixed_point( lambda K: GF_star(K, tol_TC, tol_brent_1, tol_brent, tol_pi), v_init =K_init,error_flag = 1, tol = tol_K, error_name = "pricing")

	#K_star = 5.700222761535536
	#S_bar_star = 6.534672003614626

	#rho_star = TC_star(config.rho_global_old ,K_star, config.S_bar_star, tol_TC, config.grid)

	#og.K = K_star
	#og.rho_star= rho_star
	#og.S_bar_star = config.S_bar_star
	#og.solved_flag = 1 
	#og.grid = config.grid
	#end = time.time()
	#og.time_endog = end-start 

	#Path("/scratch/kq62/{}/{}/".format(model_name,settings_file)).mkdir(parents=True, exist_ok=True)

	#pickle.dump(og, open("/scratch/kq62/{}/{}/array_6a_{}_endog.mod".format(model_name, settings_file,index),"wb"))




