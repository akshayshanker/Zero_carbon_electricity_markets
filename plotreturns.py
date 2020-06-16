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

from interpolation.splines import UCGrid, CGrid, nodes, eval_linear
from spotmarkets import EmarketModel, time_operator_factory


import dill as pickle 
import config
from scipy.optimize import root

import time
import sys

import matplotlib.pyplot as plt

def plotreturns(varlist, varname):
    returns_dict = {}

    for var in varlist:


        s_supply            = var if varname == 's_supply' else .15 
        D_bar               = var if varname == 'D_bar' else 2.818
        r_s                 = var if varname == 'r_s' else 465
        r_k                 = var if varname == 'r_k' else 1400
        zeta_storage        = var if varname == 'zeta_storage' else .5
        eta_demand          = var  if varname == 'eta_demand' else .3
        mu_supply           = var   if varname == 'mu_supply' else 0.5
        K                   = var  if varname == 'K' else 6.7/mu_supply

        og = EmarketModel(s_supply          = s_supply,         #standard deviation of supply 
                            grid_size       = 100,   #grid size of storage grid
                            D_bar           = D_bar,       #demand parameter D_bar
                            r_s             = r_s,          #cost of storage capital (Billion USD/GwH).  Set basecase to 465
                            r_k             = r_k,         #cost of generation capital (Billion USD/Gw). Set base case to 1400
                            grid_size_s     = 10,     #number of supply shocks
                            grid_size_d     = 3,    #number of demand shocks
                            zeta_storage    = zeta_storage,  # Base case is .5
                            eta_demand      = eta_demand,
                            mu_supply       = mu_supply
                            
                        )

        
        K_init                  = K #exogenous level of capital stock 

        
        GF_star, TC_star, G, F      = time_operator_factory(og) #import functions
        config.rho_global_old       = np.tile(np.ones(og.grid_size)*100, (og.grid_size_s*og.grid_size_d,1)) #initialize initial value of initial price guess
        config.S_init               = 200 
        config.grid_old     =  np.linspace(og.grid_min_s,200, og.grid_size) 

        #set tolerances 
        tol_TC      = 1e-4 #tol for pricing function iteration 
        tol_brent   = 1e-3 #second stage tol of brent's method to find S_bar
        tol_brent_1 = .5 #first stage tol of brent's method to find S_bar
        tol_pi      = 4 # number of digits to round the delta storage to determine whether it binds
        #S_bar = 7.135800156314936 
        tol_K       = .01 #tol for value of eqm. K
        config.toggle_GF =0 #set initial stage to use first stage brent tol 


        """
        Run model with exogneous stock of capital. 
        Save as model_a
        """
        
        start           = time.time()

        G_star          = lambda S_bar:  G(K_init, S_bar, tol_TC, tol_pi)

        S_range        =    np.linspace(.01, 75, 40)

        returns         = np.zeros(len(S_range))

        for i in range(len(S_range)):
            
            returns[i] = G_star(S_range[i]) + og.r_s
            #config.rho_global_old   = np.tile(np.ones(og.grid_size)*100, (og.grid_size_s*og.grid_size_d,1))


        returns_dict[var] = returns
        
        plt.plot(S_range, np.log(returns))

    plt.axhline(y = np.log(465)) 

    leglist = [ '{}= {}'.format(varname, x) for x in varlist ]
    leglist.append('Current price (USD 465 M/GwH)')

    plt.legend(leglist)
    plt.ylabel('Log marginal return (USD Million/GwH)')
    plt.xlabel('Storage capacity (GwH)')

    plt.savefig('returns_vary_{}.png'.format(varname))

    return returns_dict



#Initialize demand shock 


if __name__ == '__main__':



   varlist = [.05,.1, .2, .3]
   varname = 's_supply'
   returns_dict =plotreturns(varlist, varname)
   pickle.dump(returns_dict, open("/scratch/kq62/returns_{}.ret".format(varname),"wb"))


