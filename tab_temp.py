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

resmod0 = runres("{}".format("a"))
resmod1 = runres("{}".format("a_endog"))





mod0 = ["No storage", "%.2f (%.2f)"%(resmod0['mean_generation'], resmod0['var_generation']),resmod0["S_bar_star"], "%.2f (%.2f)"%(resmod0['mean_price'],resmod0['var_price'] ), "%.2f (%.2f)"%(resmod0['mean_demand'],resmod0['var_demand'] ),"%.2f (%.2f)"%(resmod0['mean_stor'],resmod0['var_stor'] ), "%.2f/ %.2f"%(resmod0['r_s_star'], resmod0['r_k_star'])]
mod1 = ["Baseline", "%.2f (%.2f)"%(resmod1['mean_generation'], resmod1['var_generation']),resmod1["S_bar_star"], "%.2f (%.2f)"%(resmod1['mean_price'],resmod1['var_price'] ), "%.2f (%.2f)"%(resmod1['mean_demand'],resmod1['var_demand'] ),"%.2f (%.2f)"%(resmod1['mean_stor'],resmod1['var_stor'] ), "%.2f/ %.2f"%(resmod1['r_s_star'], resmod1['r_k_star']) ]


cmod0 = ["No storage", resmod0['cov_zd'][0,1]]
cmod1 = ["Baseline", resmod1['cov_zd'][0,1]]


header = ["Gen.", "S", "Pr.", "Dem.", "Str.", "P_S| P_K"]


table= [mod0, mod1]

print(tabulate(table, headers = header,  tablefmt="latex_booktabs", floatfmt=".2f"))
