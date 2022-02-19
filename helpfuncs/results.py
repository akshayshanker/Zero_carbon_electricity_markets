import numpy as np
from interpolation import interp
#from scipy.optimize import brentq
from quantecon.optimize.scalar_maximization import brent_max
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from itertools import product
from numba import njit, prange,jit
from pathos.multiprocessing import ProcessingPool
from fixedpoint import fixed_point
from pathlib import Path


from interpolation.splines import UCGrid, CGrid, nodes, eval_linear

import dill as pickle
import config



"""
Unpacks model policy functions and convets to timeseries results and plots
"""


def  runres(model_name, sim_name,key, resolve, tol_pi):


	og = pickle.load(open("/scratch/kq62/{}.mod".format(model_name  + '/' + sim_name + '/' + key),"rb"))
	beta, delta_storage, zeta_storage       = og.beta, og.delta_storage, og.zeta_storage
	pr, p_inv                				= og.pr, og.p_inv
	grid_size, grid_min_s, grid_size_s, grid_size_d = og.grid_size, og.grid_min_s, og.grid_size_s, og.grid_size_d
	D_bar 									= og.D_bar
	P, P_supply, P_dem                      = og.P, og.P_supply, og.P_dem
	X_dem, X_supply, shock_X                = og.X_dem, og.X_supply, og.shock_X
	alpha, iota, alpha2, iota2              = og.alpha, og.iota, og.alpha2, og.iota2
	r_s, r_k                                = og.r_s, og.r_k
	TS_length                               = og.TS_length
	D_bar                                   = og.D_bar
	eta_demand                              = og.eta_demand


	K = og.K
	S_bar = og.S_bar_star
	rho_star = og.rho_star
	grid = og.grid

	rho_func = lambda e,s: np.interp(s, og.grid,og.rho_star[e]) # prob not a good idea to call e the argument here 


	def delta_func(e,s):
		#e is shock index tuble 
		#s is storage value
		price = rho_func(e,s)
		z = og.shock_X[e,0] #generator shock 
		dem_shock = og.shock_X[e,1] #demand shock 
		demand = og.p_inv(dem_shock,price, D_bar)
		gen = z*K
		return gen - demand 



	T= int(1e6)
	time = np.arange(0,T,1)
	price, d, s, gen, delta_s_eqm, delta_s_bar, s_eqm= np.zeros(T),np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
	DST                      = np.zeros(T)
	integrand_k = np.zeros(T)
	e, z = np.zeros(T),np.zeros(T)

	shock_index = np.arange(len(shock_X))
	np.random.seed(30)
	shocks = np.random.choice(shock_index, T, p=P)
	s[0] = S_bar/2
	s_eqm[0] = S_bar/2

	@njit
	def nextstor(s,d, gen):
		s1 = min([max([grid_min_s,(1-delta_storage)*s - d + gen]),S_bar])
		s2 = max([min([zeta_storage*S_bar + (1-delta_storage)*s, s1]), - zeta_storage*S_bar +(1-delta_storage)*s])
		return s2


	for i in range(T):
		z[i] = shock_X[shocks[i],0]
		e[i] = shock_X[shocks[i],1]
		price[i] = rho_func(shocks[i], s[i])
		d[i] = og.p_inv(e[i],price[i], D_bar)
		gen[i]= z[i]*og.K
		integrand_k[i] = rho_func(shocks[i], s[i])*z[i]


		if i<T-1:
			s[i+1] = nextstor(s[i], d[i], gen[i])
			s_eqm[i+1] =(1-og.delta_storage)*s_eqm[i] - d[i] + gen[i]
			delta_s_bar[i] = s[i+1] - (1- og.delta_storage)*s[i]
			delta_s_eqm[i] =  gen[i]- d[i] 
			DST[i] = (1-delta_storage)*s[i] - s[i+1]
			

	# generate storage price

	rang = np.arange(T-1)
	pool_two = ProcessingPool()

	@njit
	def return_pi(DS, S1, PI_bar):
		if round(DS, tol_pi)>= round(zeta_storage*S_bar, tol_pi):
			return - zeta_storage*PI_bar

		if -round(DS, tol_pi) >= round(zeta_storage*S_bar, tol_pi):
			return  zeta_storage*PI_bar

		if round(S1, tol_pi) >= round(S_bar, tol_pi):
			return PI_bar
		else:
			return 0
	@njit
	def T2(i):
		integrand = np.zeros(len(shock_X[:,0]))
		for j in range(len(shock_X[:,0])):
			integrand[j] = np.interp(s[i+1],grid,rho_star[j])
		Eprice = beta*(1-delta_storage)*np.dot(P,integrand)
		PI_bar = - price[i] + Eprice
		DS =(1-delta_storage)*s[i] - s[i+1]
		PI_hat = return_pi(DS,s[i+1], PI_bar)
		return  max([0,PI_hat])

	PI_hat = np.array(pool_two.map(T2, rang))

	print(np.mean(PI_hat)*(1/(1-beta)))

	results = {}

	results['S_bar_star'] = S_bar
	results['K'] = og.K
	results['delta_s_eqm'] = delta_s_eqm
	results['delta_s_bar'] = delta_s_bar

	results['r_s_star'] = (1/(1-og.beta))*np.mean(PI_hat)
	results['r_k_star'] = (1/(1-og.beta))*np.mean(integrand_k)

	results['model'] = og
	results['mean_price'] = np.mean(price)
	results['mean_stor'] = np.mean(s)
	results['mean_demand'] = np.mean(d[list(np.where(d<1000))])
	results['mean_supply'] = np.mean(z)
	results['mean_generation'] = np.mean(gen)

	results['var_price']= np.std(price)
	results['var_stor'] = np.std(s)
	results['var_demand'] = np.std(d)
	results['var_supply'] = np.std(z)
	results['var_generation'] = np.std(gen)

	results['cov_zd'] = np.corrcoef(z,d)
	results['cov_pd'] = np.corrcoef(price,d)
	results['cov_pz'] = np.corrcoef(price,z)
	results['cov_sz'] = np.corrcoef(s,z)

	results['generation'] = gen
	results['price'] = price
	results['demand'] = d
	results['stored'] = s
	results['stored_eqm'] = s_eqm
	results['demshock'] = e
	results['stockout'] =  np.isclose(s,og.grid_min_s, 1e-3).sum()/T
	results['PI_hat'] = np.mean(PI_hat)



	T= 168
	time = np.arange(0,T,1)
	price, d, s, gen = np.zeros(T),np.zeros(T), np.zeros(T), np.zeros(T)
	integrand = np.zeros(T)
	e, z = np.zeros(T),np.zeros(T)

	shock_index = np.arange(len(shock_X))
	shocks = np.random.choice(shock_index, T, p=P)
	s[0] = S_bar

	for i in range(T):
		z[i] = shock_X[shocks[i],0]
		e[i] = shock_X[shocks[i],1]
		price[i] = rho_func(shocks[i], s[i])
		d[i] = og.p_inv(e[i],price[i], D_bar)
		gen[i]= z[i]*K

		if i<T-1:
			s[i+1] = nextstor(s[i], d[i], gen[i])

	integrand[i] = rho_func(shocks[i], s[i])*z[i]

	f, axarr = plt.subplots(2,2)
	axarr[0,0].plot(time, price,  linewidth=.6)
	axarr[0,0].set_ylabel('Price (MUSD)', fontsize = 10)
	axarr[1,0].plot(time, d,  linewidth=.6)
	axarr[1,0].set_ylabel('Eqm. demand (Gw)', fontsize = 10)
	axarr[0,1].plot(time, s,  linewidth=.6)
	axarr[0,1].set_ylabel('Stored power (GwH)', fontsize = 10)
	axarr[1,1].plot(time, gen,  linewidth=.6)
	axarr[1,1].set_ylabel('Generation (Gw)', fontsize = 10)
	f.tight_layout()


	Path("Results/{}/".format(model_name  + '/' + sim_name))\
										.mkdir(parents=True, exist_ok=True)

	
	plt.savefig("Results/{}_sim.png".format(model_name  + '/' + sim_name + '/' + key))

	plt.close()

	grid_S = np.linspace(og.grid_min_s, S_bar, 500)

	shock_index2 = shock_index.reshape(og.grid_size_s,og.grid_size_d)

	f_2, axrr_2= plt.subplots(og.grid_size_s, og.grid_size_d)
	for i in range(0, og.grid_size_d):
		for j in range(0, og.grid_size_s):
			e = shock_index2[j,i]
			axrr_2[j,i].plot(grid_S, delta_func(e,grid_S))
			axrr_2[j,i].plot(grid_S, og.zeta_storage*np.ones(np.shape(grid_S))*S_bar)
			axrr_2[j,i].plot(grid_S, -og.zeta_storage*np.ones(np.shape(grid_S))*S_bar)

	f_2.tight_layout()

	plt.savefig("Results/{}_deltas.png".format(model_name  + '/' + sim_name + '/' + key))

	f_3, axrr_3= plt.subplots(og.grid_size_s, og.grid_size_d)
	for i in range(0, og.grid_size_d):
		for j in range(0, og.grid_size_s):
			e = shock_index2[j,i]
			axrr_3[j,i].plot(grid_S, rho_func(e,grid_S))

	f_3.tight_layout()

	plt.savefig("Results/{}_price.png".format(model_name  + '/' + sim_name + '/' + key))

	return results

