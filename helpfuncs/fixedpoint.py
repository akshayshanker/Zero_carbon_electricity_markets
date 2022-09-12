import numpy as np
from numba import njit, prange, jit


@njit
def fixed_point(T,\
                v_init,\
                args = (), 
                error_flag =1,\
                tol = 1e-5,\
                error_name = "unnamed",\
                maxiter = 500):
    error = 1
    iterno = 1
    v = v_init
    while error> tol and iterno<maxiter: 
        v_updated = T(v, *args)
        error = np.max(np.abs(v_updated- v))
        v = v_updated
        #print(error)
        #if error_flag ==1:
        #    print("Error for iteration %s of %s operator is %s"%(iterno, error_name, error))
        iterno = iterno +1
    return v
