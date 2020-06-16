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

import ray 

from scipy.optimize import root

import time
import sys




#@njit
def pr(e,x, D_bar):
    "Inverse of the demand function"
    eta_demand = 1/2
    xbar = x
    if xbar>0:
        price = (D_bar**(1/eta_demand))*np.exp(e/eta_demand)*xbar**(-1/eta_demand)
    if xbar< 0:
        price = np.inf
    return price

@njit
def p_inv(e,x, D_bar):
    "The demand function. Gives demand for price and demand shock"
    eta_demand = 1/2
    return D_bar*np.exp(e)*(x**(-eta_demand))



@njit
def phi_prime(x):
    "Derivative of the cost function"
    return 2

@njit
def Xi(x, eta1, eta2, eta3):
    "roundtrip function*(s-s')"
    return ((eta1*x)/(1 + np.exp(eta2*x))) + eta3*x



@njit
def Xi_prime(x, eta1, eta2, eta3):
    "Derivative of roundtrip function"
    return (eta1/(1+np.exp(eta2*x)))*(1 + (x*eta2*np.exp(eta2*x))/(1+ np.exp(eta2*x))) + eta3


#Initialize demand shock 


if __name__ == '__main__':
  

    og = EmarketModel(s_supply = np.float(sys.argv[1]),         #standard deviation of supply 
                        grid_size = 200,   #grid size of storage grid
                        grid_max_x = 100,   #initial max storge (this is redundant)
                        D_bar = .344,       #demand parameter D_bar
                        r_s = np.float(sys.argv[2]),          #cost of storage capital (Billion USD/GwH).  Set basecase to 465
                        r_k = np.float(sys.argv[3]),         #cost of generation capital (Billion USD/Gw). Set base case to 1400
                        grid_size_s =3,     #number of supply shocks
                        grid_size_d = 3,    #number of demand shocks
                        p=pr,               #define  demand function
                        p_inv=p_inv,        #define inverse demand function 
                        Xi = Xi,            #redundant for now
                        Xi_prime = Xi_prime, #redundant for now
                        zeta_storage = np.float(sys.argv[4]),  # Base case is .5
                    )

    
    K_init = 6.7/np.inner(og.P_supply, og.X_supply[0:og.grid_size_s]) #exogenous level of capital stock 

    
    GF_star, TC_star, G = time_operator_factory(og) #import functions
    config.grid_old = og.grid #initial grid on config file 
    config.rho_global_old = np.tile(np.ones(og.grid_size)*100, (og.grid_size_s*og.grid_size_d,1)) #initialize initial value of initial price guess
    config.S_init = 200 

    #set tolerances 
    tol_TC = 1e-4 #tol for pricing function iteration 
    tol_brent = 1e-3 #second stage tol of brent's method to find S_bar
    tol_brent_1 = .5 #first stage tol of brent's method to find S_bar
    tol_pi = 1e-8 #tol for evaluatiing characteristic function of boundary conditions 
    #S_bar = 7.135800156314936 
    tol_K = .01 #tol for value of eqm. K
    config.toggle_GF =0 #set initial stage to use first stage brent tol 


    """
    Run model with exogneous stock of capital. 
    Save as model_a
    """
    
    start = time.time()

    G_star= lambda S_bar:  G(K_init, S_bar, tol_TC, tol_pi)
    S_bar_star = brentq(G_star, 1, 100, xtol = tol_brent)
    
    ray.init()
    rho_star = TC_star(config.rho_global_old ,K_init, S_bar_star, tol_TC)
    ray.shutdown()

    og.K = K_init
    og.rho_star= rho_star
    og.S_bar_star = S_bar_star
    og.solved_flag = 1 
    og.grid = config.grid
    end = time.time()
    og.time_exog = end-start 


    pickle.dump(og, open("{}.mod".format(sys.argv[5]),"wb"))

    """
    Run model with endogenous stock of capital. Save as model a_prime
    """
    
    config.toggle_GF =0
    start = time.time()
    K_star = fixed_point( lambda K: GF_star(K, tol_TC, tol_brent_1, tol_brent, tol_pi), v_init =K_init,error_flag = 1, tol = tol_K, error_name = "pricing")

    #K_star = 5.700222761535536
    #S_bar_star = 6.534672003614626
    ray.init()
    rho_star = TC_star(config.rho_global_old ,K_star, config.S_bar_star, tol_TC)
    ray.shutdown()

    og.K = K_star
    og.rho_star= rho_star
    og.S_bar_star = config.S_bar_star
    og.solved_flag = 1 
    og.grid = config.grid
    end = time.time()
    og.time_endog = end-start 



    pickle.dump(og, open("{}_endog.mod".format(sys.argv[5]),"wb"))




