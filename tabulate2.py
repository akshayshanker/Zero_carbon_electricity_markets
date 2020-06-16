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
from tabulate import tabulate

from results_2 import runres

import matplotlib.pyplot as plt


import dill as pickle

resmod0 = runres("{}".format("a_endog"))
resmod1 = runres("{}".format("c_endog"))
resmod2  = runres("{}".format("j_endog"))

resmod3 = runres("{}".format("b_endog"))
resmod4 = runres("{}".format("m_endog"))
resmod5  = runres("{}".format("n_endog"))

resmod5  = runres("{}".format("e"))



mod0 = ["Base price", "%.2f (%.2f)"%(resmod0['mean_generation'], resmod0['var_generation']),resmod0['K'],resmod0["S_bar_star"], "%.2f (%.2f)"%(resmod0['mean_price'],resmod0['var_price'] ), "%.2f (%.2f)"%(resmod0['mean_demand'],resmod0['var_demand'] ),"%.2f (%.2f)"%(resmod0['mean_stor'],resmod0['var_stor'] ), resmod0['stockout']]
mod1 = ["Low price", "%.2f (%.2f)"%(resmod1['mean_generation'],resmod1['var_generation']),resmod1['K'], resmod1["S_bar_star"], "%.2f (%.2f)"%(resmod1['mean_price'],resmod1['var_price'] ), "%.2f (%.2f)"%(resmod1['mean_demand'],resmod1['var_demand'] ),"%.2f (%.2f)"%(resmod1['mean_stor'],resmod1['var_stor'] ), resmod1['stockout'] ]
mod2 = ["Very low price", "%.2f (%.2f)"%(resmod2['mean_generation'], resmod2['var_generation']),resmod2['K'], resmod2["S_bar_star"], "%.2f (%.2f)"%(resmod2['mean_price'],resmod2['var_price'] ), "%.2f (%.2f)"%(resmod2['mean_demand'],resmod2['var_demand'] ),"%.2f (%.2f)"%(resmod2['mean_stor'],resmod2['var_stor'] ), resmod2['stockout'] ]
mod3 = ["Base price", "%.2f (%.2f)"%(resmod3['mean_generation'], resmod3['var_generation']),resmod3['K'], resmod3["S_bar_star"], "%.2f (%.2f)"%(resmod3['mean_price'],resmod3['var_price'] ), "%.2f (%.2f)"%(resmod3['mean_demand'],resmod3['var_demand'] ),"%.2f (%.2f)"%(resmod3['mean_stor'],resmod3['var_stor'] ), resmod3['stockout']]
mod4 = ["Low price", "%.2f (%.2f)"%(resmod4['mean_generation'], resmod4['var_generation']),resmod4['K'], resmod4["S_bar_star"], "%.2f (%.2f)"%(resmod4['mean_price'],resmod4['var_price'] ), "%.2f (%.2f)"%(resmod4['mean_demand'],resmod4['var_demand'] ),"%.2f (%.2f)"%(resmod4['mean_stor'],resmod4['var_stor'] ), resmod4['stockout'] ]
mod5 = ["Very low price", "%.2f (%.2f)"%(resmod5['mean_generation'], resmod5['var_generation']),resmod5['K'],resmod5["S_bar_star"], "%.2f (%.2f)"%(resmod5['mean_price'],resmod5['var_price'] ), "%.2f (%.2f)"%(resmod5['mean_demand'],resmod5['var_demand'] ),"%.2f (%.2f)"%(resmod5['mean_stor'],resmod5['var_stor'] ), resmod5['stockout'] ]

cmod0 = ["No storage", resmod0['cov_zd'][0,1]]
cmod1 = ["Baseline", resmod1['cov_zd'][0,1]]


header = ["Gen.", "S", "K" "Pr.", "Dem.", "Str.", "lowstor %"]


table= [mod0, mod1, mod2]

print(tabulate(table, headers = header,  tablefmt="latex_booktabs", floatfmt=".2f"))

restab = open("results012_tab.tex", 'w')

restab.write(tabulate(table, headers = header,  tablefmt="latex_booktabs", floatfmt=".2f"))

restab.close()

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

