import numpy as np
from spotmarkets import EmarketModel, time_operator_factory
from helpfuncs.results import runres
import dill as pickle 
import sys
from sklearn.utils.extmath import cartesian
from numpy import genfromtxt
from pathlib import Path

def plotreturns(world_comm, model_name, sim_name, parameters):
    
    returns_dict = {}
    s_supply = parameters[1]
    D_bar = parameters[6]
    r_s  = parameters[2]*1E9
    r_k  = parameters[3]*1E9
    zeta_storage  = parameters[4]
    eta_demand  = parameters[5]
    mu_supply  =  parameters[0]

    # set tolerances 
    tol_TC      = 1e-2 #tol for pricing function iteration 
    tol_pi      = 4 # number of digits to round the delta storage to determine whether it binds

    # Set the ranges of the supply capacity
    S_range =  np.linspace(0.1, 1000, 96)
    #S_range = np.insert(S_range,0, 0.1)
    S_index = np.arange(96)
    
    # Set the ranges of the gen. capacity
    K_range =  np.linspace(0.1, 500, 96)
    K_index = np.arange(96)
    
    # Set the ranges of the variance 
    VS_range = np.linspace(.01, .18, 4)
    VS_index = np.arange(4)
    
    SK_CART = cartesian([S_range, K_range ])
    SK_CART_index = cartesian([S_index, K_index])

    for j in range(len(VS_range)):

        S = SK_CART[world_comm.rank][0]
        K = SK_CART[world_comm.rank][1]
        s_supply = VS_range[j]
        color_layer_1 = world_comm.rank

        U = pickle.load(open("{}/seed_u.pkl"\
                        .format(model_name),"rb"))

        og = EmarketModel(s_supply = s_supply, # standard deviation of supply 
                    grid_size = 100,  # grid size of storage grid
                    D_bar = D_bar, # demand parameter D_bar
                    r_s = r_s, # cost of storage capital (Billion USD/GwH).  Set basecase to 465
                    r_k = r_k, # cost of generation capital (Billion USD/Gw). Set base case to 1400
                    grid_size_s = 4, # number of supply shocks
                    grid_size_d = 4, # number of demand shocks
                    zeta_storage = zeta_storage,# Base case is .5
                    eta_demand  = eta_demand,
                    s_demand = parameters[8],
                    mu_supply = mu_supply, U = U)
        
        # import functions
        GF_RMSE, F_star, TC_star, G, F= time_operator_factory(og)
                    
        stor_pnl, rho_star = G(K,S, tol_TC, tol_pi)
        gen_pn = F(K, S, tol_TC, rho_star)

        og.stor_pnl = stor_pnl
        og.gen_pn = gen_pn
        og.K  = K
        og.rho_star = rho_star
        og.S_bar_star = S
        og.solved_flag = 1 
        og.grid = np.linspace(og.grid_min_s, S, og.grid_size)

        Path("/scratch/pv33/{}/static/{}/".format(model_name,sim_name))\
                                        .mkdir(parents=True, exist_ok=True)
        
        results = runres(og, model_name, sim_name, 'key', 1, 4, plot = False)

        pickle.dump(results, open("/scratch/pv33/{}/static/{}/model_{}_{}_{}_static_res.pkl"\
                        .format(model_name,sim_name,\
                                    SK_CART_index[color_layer_1][0],\
                                    SK_CART_index[color_layer_1][1],\
                                    j),\
                                    "wb"))

    return returns_dict


if __name__ == '__main__':

    from mpi4py import MPI as MPI4py
    world = MPI4py.COMM_WORLD

    # Load settings file and set simulation name, model name 
    settings_file = sys.argv[1]
    sim_name = sys.argv[1]
    model_name = 'ERCOT_main_v_1'

    array = genfromtxt('Settings/ERCOT/{}.csv'\
            .format(settings_file), delimiter=',')[1:]
    
    # Take the baseline parameters from the baseline sim settings 
    # these are top row of parameters 
    parameters = array[0]

    # Run the static solver across each of the nodes
    # each nodes solves one point in the array of 
    # exogenous capital stocks X variance 
    # each node saves the results 

    returns_dict = plotreturns(world, model_name, sim_name, parameters)

