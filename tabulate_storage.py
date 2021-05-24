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
from results import runres
import matplotlib.pyplot as plt
import dill as pickle
from tabulate import tabulate


def tabulate_storage(modlist, filename):
		
	results = {}
	table   = []
	for key in modlist:
		print("Calculating results from {}".format(key))
		results[key] = runres("{}".format(key),1,4)
		results_row  = ["key",\
						"%.2f (%.2f)"%(results[key]['mean_generation'], results[key]['var_generation']), \
						 results[key]['K'],results[key]["S_bar_star"],\
						 "%.2f (%.2f)"%(results[key]['mean_price'], results[key]['var_price'] ),\
						 "%.2f (%.2f)"%(results[key]['mean_demand'], results[key]['var_demand'] ),\
						 "%.2f (%.2f)"%(results[key]['mean_stor'],results[key]['var_stor'] ),\
						 results[key]['stockout']]
		table.append(results_row)

	print(table)

	header = ["Gen.", "S", "K", "Pr.", "Dem.", "Str.", "lowstor %"]


	print(tabulate(table, headers = header,  tablefmt="latex_booktabs", floatfmt=".2f"))
	 
	restab = open("results_tab_{}.tex".format(filename), 'w')

	restab.write(tabulate(table, headers = header,  tablefmt="latex_booktabs", floatfmt=".2f"))

	restab.close()

	return 



if __name__ == '__main__':

	modlist = ['baseline', 'baseline_endog']
	for i in range(7):
		modlist.append('array_6_{}'.format(i))

	tabulate_storage(modlist, 'calib')

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

