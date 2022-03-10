import numpy as np
from interpolation import interp
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from numba import njit
from spotmarkets import EmarketModel, time_operator_factory
from helpfuncs.results import runres
import dill as pickle 
import config
import time
import sys
import matplotlib.pyplot as plt
from sklearn.utils.extmath import cartesian


def plotreturns(world_comm, model_name, sim_name, parameters):
    returns_dict = {}
    for var in varlist:

        s_supply = var if varname == 's_supply' else parameters[1]
        D_bar = var if varname == 'D_bar' else parameters[6]
        r_s  = var if varname == 'r_s' else parameters[2]*1E9
        r_k  = var if varname == 'r_k' else parameters[3]*1E9
        zeta_storage  = var if varname == 'zeta_storage' else parameters[4]
        eta_demand  = var  if varname == 'eta_demand' else parameters[5]
        mu_supply  = var  if varname == 'mu_supply' else parameters[0]

        og = EmarketModel(s_supply          = s_supply, # standard deviation of supply 
                            grid_size       = 100,  # grid size of storage grid
                            D_bar           = D_bar, # demand parameter D_bar
                            r_s             = r_s, # cost of storage capital (Billion USD/GwH).  Set basecase to 465
                            r_k             = r_k, # cost of generation capital (Billion USD/Gw). Set base case to 1400
                            grid_size_s     = 4, # number of supply shocks
                            grid_size_d     = 3, # number of demand shocks
                            zeta_storage    = zeta_storage,# Base case is .5
                            eta_demand      = eta_demand,
                            mu_supply       = mu_supply)

        # exogenous level of capital stock 
        K_init = K 
        # import functions
        GF_star, TC_star, G, F = time_operator_factory(og)
        # initialize initial value of initial price guess
        config.rho_global_old = np.tile(np.ones(og.grid_size)*100, (og.grid_size_s*og.grid_size_d,1)) 
        config.S_init  = 200 
        config.grid_old =  np.linspace(og.grid_min_s,200, og.grid_size) 

        #set tolerances 
        tol_TC      = 1e-2 #tol for pricing function iteration 
        tol_pi      = 4 # number of digits to round the delta storage to determine whether it binds
        #S_bar = 7.135800156314936 

        """
        Run model with exogneous stock of capital. 
        Save as model_a
        """
        start = time.time()
        #G_star = lambda S_bar:  G(K_init, S_bar, )
        S_range =  np.linspace(.01, 100, 50)
        K_range =  np.linspace(.01, 100, 40)
        returns  = np.zeros(int(len(S_range)*len(K_range)))

        SK_CART = cartesian([S_range, K_range])

        S = SK_CART[world_comm.rank][0]
        K = SK_CART[world_comm.rank][1]
                    
        stor_pnl, rho_star = G(K,S, tol_TC, tol_pi)
        gen_pn = F(K, S, tol_TC, rho_star)

        og.stor_pnl = stor_pnl
        og.gen_pn = gen_pn
        og.K  = K
        og.rho_star = rho_star
        og.S_bar_star = S
        og.solved_flag = 1 
        og.grid = config.grid

        Path("/scratch/kq62/{}/{}/".format(model_name,sim_name))\
                                        .mkdir(parents=True, exist_ok=True)
        
        pickle.dump(og, open("/scratch/kq62/{}/{}/{}_{}_static.mod"\
                                .format(model_name,sim_name,\
                                                    color_layer_1),"wb"))

        returns_dict[var] = returns
        
        plt.plot(S_range, np.log(returns))

    plt.axhline(y = np.log(r_s)) 

    leglist = [ '{}= {}'.format(varname, x) for x in varlist ]
    leglist.append('Current price (USD 465 M/GwH)')

    plt.legend(leglist)
    plt.ylabel('Log marginal return (USD Million/GwH)')
    plt.xlabel('Storage capacity (GwH)')

    plt.savefig('returns_vary_{}_e3.png'.format(varname))

    return returns_dict

#Initialize demand shock 

if __name__ == '__main__':


    # load array of parameter values for each model 
    settings_file = 'sys.argv[1]'
    array = genfromtxt('Settings/{}.csv'\
            .format(settings_file), delimiter=',')[1:]
    model_name = 'main_v_2'
    sim_name = 'static'
    parameters = array[0]


    varlist = [10, 20,30,40, 50,60, 70,80]
    varname = 'K'
    returns_dict = plotreturns(model_name, sim_name, varlist, varname, parameters)
    pickle.dump(returns_dict, open("/scratch/kq62/returns_{}.ret".format(varname),"wb"))


