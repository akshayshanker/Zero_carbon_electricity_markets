import numpy as np
from interpolation import interp
#from scipy.optimize import brentq
from quantecon.optimize.scalar_maximization import brent_max
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from itertools import product
from numba import njit, prange
from pathos.multiprocessing import ProcessingPool
from fixedpoint import fixed_point
from helpfuncs.results import runres
import matplotlib.pyplot as plt
import dill as pickle
from tabulate import tabulate
from numpy import genfromtxt


def tabulate_storage(modlist, row_names,filename, model_name, sim_name):
		
	results = {}
	table   = []
	for key, rname in zip(modlist, row_names):
		print("Calculating results from {}".format(key))
		results[key] = runres(model_name, sim_name,key, 1,4)
		results_row  = [rname,\
						"%.2f (%.2f)"%(results[key]['mean_generation'], results[key]['var_generation']), \
						 "%.2f"%results[key]['K'],"%.2f"%results[key]["S_bar_star"],\
						 "%.2f (%.2f)"%(results[key]['mean_price']*1e-3, results[key]['var_price']*1e-3 ),\
						 "%.2f (%.2f)"%(results[key]['mean_demand'], results[key]['var_demand'] ),\
						 "%.2f (%.2f)"%(results[key]['mean_stor'],results[key]['var_stor'] )]\
						 #results[key]['stockout']]
		table.append(results_row)

	print(table)

	header = ["Av gen.", "Gen. cap.", "S cap.","Pr.", "Dem.", "Av str."]#, "lowstor %"]


	print(tabulate(table, headers = header,  tablefmt="latex_booktabs", floatfmt=".2f"))
	 
	restab = open("Results/{}/results_tab.tex".format(model_name  + '/' + sim_name + '/'), 'w')

	restab.write(tabulate(table, headers = header,  tablefmt="latex_booktabs", floatfmt=".2f"))

	restab.close()

	return 

def gen_arr_storage(modlist, row_names,filename):
	results = {}
	table_s   = []
	table_k   = []
	table_v = []
	for key, rname in zip(modlist, row_names):
		print("Calculating results from {}".format(key))
		og = pickle.load(open("/scratch/kq62/main_v_2/baseline/{}.mod".format(key),"rb"))
		table_s.append(og.S_bar_star)
		table_k.append(og.K)
		table_v.append(og.s_supply)

	return table_s, table_k, table_v


if __name__ == '__main__':


	array = genfromtxt('Settings/baseline.csv', delimiter=',')[1:] 

	model_name = 'main_v_2'
	sim_name = 'baseline_1'
	settings_file = 'baseline_1'

	array = genfromtxt('Settings/{}.csv'\
			.format(settings_file), delimiter=',')[1:]

	modlist = []
	row_names = []

	for i in range(len(array)):
		modlist.append('{}_{}_endog'.format(sim_name,i))
		row_names.append(array[i, -1])

	#tabulate_storage(modlist, row_names, 'baselines', model_name, sim_name)

	     
	table_s, table_k, table_v =  gen_arr_storage(modlist, row_names,'calib')
	plt.close()
	plt.plot(table_v,table_s, label = 'Storage', color= 'red')
	plt.plot(table_v,table_k, label = 'Capital', color= 'blue')
	plt.xlabel('Variance of supply')
	plt.legend()
	plt.savefig('sotage_variance.png')




""""

table2 = [mod3, mod4, mod5]
print(tabulate(table2, headers = header,  tablefmt="latex_booktabs", floatfmt=".2f"))
restab = open("results234_tab.tex", 'w')
restab.write(tabulate(table2, headers = header,  tablefmt="latex_booktabs", floatfmt=".2f"))
restab.close()


#table3 = [cmod0, cmod1, cmod2, cmod3, cmod4]

print(tabulate(table3, headers = header,  tablefmt="latex_booktabs", floatfmt=".2f"))
restab = open("resultscov_tab.tex", 'w')
restab.write(tabulate(table3, headers = header,  tablefmt="latex_booktabs", floatfmt=".2f"))
restab.close()




swe_price =  np.genfromtxt('price_swe.csv', delimiter=',')

f, axarr = plt.subplots(1,3, sharey='row')
axarr[0].boxplot(np.log(swe_price[1:,]))
axarr[0].set_xlabel('Swedish prices today', fontsize = 10)
axarr[0].set_ylabel('Log of price')
axarr[1].boxplot(np.log(resmod0['price']))
axarr[1].set_xlabel('Current batt.price/ low var.', fontsize = 10)
axarr[2].boxplot(np.log(resmod3['price']))
axarr[2].set_xlabel('Low batt. price/ low var', fontsize = 10)
f.tight_layout()


plt.savefig("/home/akshay_shanker/model_box_price.png")


f, axarr = plt.subplots(1,2, sharey='row')
axarr[0].boxplot(np.log(resmod0['stored']))
axarr[0].set_xlabel('Current batt. price/ low variance', fontsize = 10)
axarr[0].set_ylabel('Log of stored power')
axarr[1].boxplot(np.log(resmod3['stored']))
axarr[1].set_xlabel('Low batt. price/ low variance', fontsize = 10)
f.tight_layout()


plt.savefig("/home/akshay_shanker/model_box_stor.png")

swe_dem =  np.genfromtxt('dem_swe.csv', delimiter=',')

f, axarr = plt.subplots(1,3, sharey='row')
axarr[0].boxplot(np.log(swe_dem[1:,]))
axarr[0].set_xlabel('Swedish demand today', fontsize = 10) 
axarr[0].set_ylabel('Log of eqm. demand')
axarr[1].boxplot(np.log(resmod0['demand']))
axarr[1].set_xlabel('Current batt. price/ low var', fontsize = 10)
axarr[2].boxplot(np.log(resmod3['demand']))
axarr[2].set_xlabel('Low batt. price/ low var', fontsize = 10)  

f.tight_layout()


plt.savefig("/home/akshay_shanker/model_box_dem.png")


f, axarr = plt.subplots(1,3, sharey='row')
axarr[0].boxplot(np.log(swe_dem[1:,]))
axarr[0].set_xlabel('Swedish demand today', fontsize = 10) 
axarr[0].set_ylabel('Log of eqm. demand')
axarr[1].boxplot(np.log(resmod0['demand']))
axarr[1].set_xlabel('Current batt. price/ low var', fontsize = 10)
axarr[2].boxplot(np.log(resmod3['demand']))
axarr[2].set_xlabel('Low batt. price/ low var', fontsize = 10)  

f.tight_layout()

plt.clf()
plt.plot(og.grid, og.p_inv(-og.shock_X[2,1],og.rho_star[2])+ og.K*og.shock_X[2,0])
plt.plot(og.grid, og.p_inv(-og.shock_X[3,1],og.rho_star[3])+ og.K*og.shock_X[3,0])
"""

