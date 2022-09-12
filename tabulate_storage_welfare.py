"""
Script loads a Equilibrium Spotmarkets simulation group and:
-- tabulates main results in each individual simulation
-- plots generation and welfare on storage price and supply variance 

Script is run on single core via IPython interface 

"""


import numpy as np
import matplotlib.pyplot as plt
from helpfuncs.results import runres
import matplotlib.pyplot as plt
import dill as pickle
from numba import njit
from tabulate import tabulate
from numpy import genfromtxt

from scipy.interpolate import UnivariateSpline as spline
from scipy.interpolate import interp1d



@njit 
def u_inv(e,W,D_bar, eta_demand):

    """"
    Utility function
    """
    theta = 1/eta_demand

    return ((W*D_bar ** (-theta))*(1-theta)) ** (1/(1-theta))



def tabulate_storage(comm,modlist, row_names, filename, model_name, sim_name, tab = False):

    results = {}
    table = []
    table_rs = []
    table_u = []
    table_k = []
    table_s = []
    table_v = []

    #stor_list = zip(modlist, row_names)

    key = modlist[comm.rank]
    rname = row_names[comm.rank]

    # Unpack results, make table
    print("Calculating results from {}".format(key))
    og = pickle.load(
        open("/scratch/tp66/main_v_2/{}/{}.mod".format(sim_name, key), "rb"))
    
    results = runres(og,model_name, sim_name, key, 1, 4, plot= False)
    
    if tab == True:

        CEGW = u_inv(0,results['WF']*(1-og.beta), og.D_bar, og.eta_demand)

        results_row = [comm.rank,
            "%.2f/%.2f"%(og.r_k*1E-09,og.r_s*1E-09),
            "%.2f" %
            results['K'],
            "%.2f" %
            results["S_bar_star"],
            "%.2f (%.2f)" %
            (results['mean_generation'],
             results['var_generation']),
            "%.2f (%.2f)" %
            (results['mean_price'] *
             1e-3,
             results['var_price'] *
             1e-3),
            "%.2f (%.2f)" %
            (results['mean_demand'],
             results['var_demand']),
            "%.2f (%.2f)" %
            (results['mean_stor'],
             results['var_stor']),
            "%.2f " %
            CEGW]

        #results = comm.gather(results, root=0)

        table = results_row

        table = comm.gather(table, root=0)


        if comm.rank == 0:
            header = ["no.","K/S Price",
                "Gen. cap.",
                "S cap.",
                "Av gen.",
                "Pr.",
                "Dem.",
                "Av str.",
                "CEGW"]  # , "lowstor %"]
            restab = open(
                "Results/{}results_tab_welfare.tex".format(model_name + '/' + sim_name + '/'), 'w')
            restab.write(
                tabulate(
                    table,
                    headers=header,
                    tablefmt="latex_booktabs",
                    floatfmt=".2f"))
            restab.close()

    
    table_rs = og.r_s
    table_u = results['WF']
    table_k = og.K
    table_s = og.S_bar_star
    table_v = og.s_supply

    
    table_rs = np.array(comm.gather(table_rs, root=0))
    table_u = np.array(comm.gather(table_u, root=0))
    table_k = np.array(comm.gather(table_k, root=0))
    table_s = np.array(comm.gather(table_s, root=0))
    table_v = np.array(comm.gather(table_v, root=0))
    


    return table_rs, table_u, table_k, table_s, table_v,results


if __name__ == '__main__':
    import sys

    model_name = 'main_v_2'

    from mpi4py import MPI as MPI4py
    comm = MPI4py.COMM_WORLD
    import seaborn as sns

    # Unpack all 
    table_rs = {}
    table_u = {}
    table_k = {}
    table_s = {}
    table_v = {}
    results = {}

    #'array_1_rs', 'array_2_rs','array_1', 'array_2', 

    for sim_name in ['welfare']:

        settings_file = sim_name

        array = genfromtxt('Settings/{}.csv'
                           .format(settings_file), delimiter=',')

        #array = [0]
        modlist = []
        row_names = []

        for i in range(len(array)):
            modlist.append('{}_{}_endog'.format(sim_name, i))
            row_names.append(array[i, -1])

        table_rs[sim_name], table_u[sim_name], table_k[sim_name], table_s[sim_name], table_v[sim_name], results[sim_name] = tabulate_storage(
            comm, modlist, row_names, 'baselines', model_name, sim_name, tab = True)

        comm.Barrier()
