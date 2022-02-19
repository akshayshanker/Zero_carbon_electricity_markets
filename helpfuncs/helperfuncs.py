import numpy as np
from numba import njit

@njit
def interp_as(xp,yp,x, extrap = True):

    """Function  interpolates 1D
    with linear extraplolation 

    Parameters
    ----------
    xp : 1D array
          points of x values
    yp : 1D array
          points of y values
    x  : 1D array
          points to interpolate 

    Returns
    -------
    evals: 1D array  
            y values at x 

    """

    evals = np.zeros(len(x))
    if extrap == True and len(xp)>1:
        for i in range(len(x)):
            if x[i]< xp[0]:
                if (xp[1]-xp[0])!=0:
                    evals[i]= yp[0]+(x[i]-xp[0])*(yp[1]-yp[0])\
                        /(xp[1]-xp[0])
                else:
                    evals[i] = yp[0]

            elif x[i] > xp[-1]:
                if (xp[-1]-xp[-2])!=0:
                    evals[i]= yp[-1]+(x[i]-xp[-1])*(yp[-1]-yp[-2])\
                        /(xp[-1]-xp[-2])
                else:
                    evals[i] = yp[-1]
            else:
                evals[i]= np.interp(x[i],xp,yp)
    else:
        evals = np.interp(x,xp,yp)
    return evals
