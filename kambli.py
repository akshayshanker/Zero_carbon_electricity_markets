""""
Module contains jitted stochastic grid search opimization
functiins
"""
import numpy as np
from numba import njit

@njit
def rand_grid_generator(cardin_X
					):
	 
	
	return  np.float16(np.random.uniform(0,1,cardin_X))  
	