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

    index = int(sys.argv[1])
    array = genfromtxt('array_6.csv', delimiter=',')[1:]      
    parameters = array[index]

    print(parameters)
  
    og = EmarketModel(s_supply      = parameters[0],         #standard deviation of supply 
                        grid_size   = 150,                #grid size of storage grid
                        grid_max_x  = 100,               #initial max storge (this is redundant)
                        D_bar       = .344,                   #demand parameter D_bar
                        r_s         = parameters[1],            #cost of storage capital (Billion USD/GwH).  Set basecase to 465
                        r_k         = parameters[2],            #cost of generation capital (Billion USD/Gw). Set base case to 1400
                        grid_size_s = 15,               #number of supply shocks
                        grid_size_d = 3,                #number of demand shocks
                        zeta_storage = parameters[3],   # Base case is .5
                        eta_demand   = .1
                        
                    )

    
    K_init                  = 6.7/np.inner(og.P_supply, og.X_supply[0:og.grid_size_s]) #exogenous level of capital stock 

    
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
    S_bar_star      = brentq(G_star, 1e-1, 2500, xtol = tol_brent)
    
    rho_star        = TC_star(config.rho_global_old ,K_init, S_bar_star, tol_TC, config.grid)

    og.K            = K_init
    og.rho_star     = rho_star
    og.S_bar_star   = S_bar_star
    og.solved_flag  = 1 
    og.grid         = config.grid
    end             = time.time()
    og.time_exog    = end-start 


    pickle.dump(og, open("/scratch/kq62/array_6_{}.mod".format(sys.argv[1]),"wb"))

    

    """
    Run model with endogenous stock of capital. Save as model a_prime
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


    pickle.dump(og, open("/scratch/kq62/array_6_{}_endog.mod".format(sys.argv[1]),"wb"))




