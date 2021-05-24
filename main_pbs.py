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
from numpy import genfromtxt






#Initialize demand shock 


if __name__ == '__main__':
    from mpi4py import MPI as MPI4py
    world = MPI4py.COMM_WORLD

    index = int(world.Get_rank())
    array = genfromtxt('Settings/array_6.csv', delimiter=',')[1:]      
    parameters = array[index]

    print(parameters)
  
    og = EmarketModel(s_supply       = parameters[1],    #variance deviation of supply 
                        mu_supply    = parameters[0],
                        grid_size    = 100,              #grid size of storage grid
                        grid_max_x   = 100,              #initial max storge (this is redundant)
                        D_bar        = parameters[6],    #demand parameter D_bar
                        r_s          = parameters[2]*1E9,    #cost of storage capital (USD/GwH).  Set basecase to 465
                        r_k          = parameters[3]*1E9,    #cost of generation capital (USD/Gw). Set base case to 1400
                        grid_size_s  = 10,               #number of supply shocks
                        grid_size_d  = 5,                #number of demand shocks
                        zeta_storage = parameters[4],   # Base case is .5
                        eta_demand   = parameters[5]
                        
                    )

    
    K_init                  = 39.8/np.inner(og.P_supply, og.X_supply[0:og.grid_size_s]) #exogenous level of capital stock 

    
    GF_star, TC_star, G, F  = time_operator_factory(og) #import functions
    config.grid_old         = og.grid #initial grid on config file 
    config.rho_global_old   = np.tile(np.ones(og.grid_size)*100, (og.grid_size_s*og.grid_size_d,1)) #initialize initial value of initial price guess
    config.S_init           = 200 

    #set tolerances 
    tol_TC      = 1e-4 #tol for pricing function iteration 
    tol_brent   = 1e-3 #second stage tol of brent's method to find S_bar
    tol_brent_1 = .5 #first stage tol of brent's method to find S_bar
    tol_pi      = parameters[4] # number of digits to round the delta storage to determine whether it binds
    #S_bar = 7.135800156314936 
    tol_K       = .01 #tol for value of eqm. K
    config.toggle_GF =0 #set initial stage to use first stage brent tol 


    """
    Run model with exogneous stock of capital. 
    Save as model_a
    """
    
    start           = time.time()

    G_star          = lambda S_bar:  G(K_init, S_bar, tol_TC, tol_pi)
    S_bar_star      = brentq(G_star, 1e-10, 2000, xtol = tol_brent)
    
    rho_star        = TC_star(config.rho_global_old ,K_init, S_bar_star, tol_TC, config.grid)

    og.K            = K_init
    og.rho_star     = rho_star
    og.S_bar_star   = S_bar_star
    og.solved_flag  = 1 
    og.grid         = config.grid
    end             = time.time()
    og.time_exog    = end-start 


    pickle.dump(og, open("/scratch/kq62/array_6a_{}.mod".format(index),"wb"))


    """
    Run model with endogenous stock of capital. 
    """
    
    config.toggle_GF =0
    start = time.time()
    K_star = fixed_point( lambda K: GF_star(K, tol_TC, tol_brent_1, tol_brent, tol_pi), v_init =K_init,error_flag = 1, tol = tol_K, error_name = "pricing")

    #K_star = 5.700222761535536
    #S_bar_star = 6.534672003614626

    rho_star = TC_star(config.rho_global_old ,K_star, config.S_bar_star, tol_TC, config.grid)

    og.K = K_star
    og.rho_star= rho_star
    og.S_bar_star = config.S_bar_star
    og.solved_flag = 1 
    og.grid = config.grid
    end = time.time()
    og.time_endog = end-start 


    pickle.dump(og, open("/scratch/kq62/array_6a_{}_endog.mod".format(index),"wb"))




