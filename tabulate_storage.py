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
from tabulate import tabulate
from numpy import genfromtxt
from pathlib import Path


from scipy.interpolate import UnivariateSpline as spline
from scipy.interpolate import interp1d


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
        open("/scratch/tp66/ERCOT_main_v_1/{}/{}.mod".format(sim_name, key), "rb"))
    
    results = runres(og,model_name, sim_name, key, 1, 4, plot= False)
    
    if tab == True:
        results_row = [
            rname,
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
             results['var_stor'])]
            #"%.2f " %
            #results['WF']]

        #results = comm.gather(results, root=0)

        table = results_row

        table = comm.gather(table, root=0)


        Path("Results/{}".format(model_name + '/' + sim_name + '/'))\
                                        .mkdir(parents=True, exist_ok=True)
        if comm.rank == 0:
            header = [
                "Gen. cap.",
                "S cap.",
                "Av gen.",
                "Pr.",
                "Dem.",
                "Av str."]  # , "lowstor %"]
            restab = open(
                "Results/{}/results_tab.tex".format(model_name + '/' + sim_name), 'w')
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
    results = comm.gather(results, root=0)
    table_k = np.array(comm.gather(table_k, root=0))
    table_s = np.array(comm.gather(table_s, root=0))
    table_v = np.array(comm.gather(table_v, root=0))
    


    return table_rs, table_u, table_k, table_s, table_v,results


if __name__ == '__main__':
    import sys

    model_name = 'ERCOT_main_v_1'

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
    row_names = ['Baseline', 'High intermittency', 'Low intermittency ', 'Low str. cap. price']

    for sim_name in ['baseline_3','baseline_05','baseline_1', 'baseline_nostor']:
    #for sim_name in ['array_2', 'array_1']:

        settings_file = sim_name

        array = genfromtxt('Settings/ERCOT/{}.csv'
                           .format(settings_file), delimiter=',')[1:]

        modlist = []
        #row_names = []

        for i in range(len(array)):
            modlist.append('{}_{}_endog'.format(sim_name, i))
            row_names.append(array[i, -1])

        table_rs[sim_name], table_u[sim_name], table_k[sim_name], table_s[sim_name], table_v[sim_name], results[sim_name] = tabulate_storage(
            comm, modlist, row_names, 'baselines', model_name, sim_name, tab = True)
        print(table_s[sim_name])

        comm.Barrier()

    """
    # Capital on variance

    if comm.rank ==0:

        col_dict = [np.array([227, 27, 35])/255, np.array([0, 45, 106])/255]

        plt.close()
        for sim_name, linename,col in zip(['array_1', 'array_2'], ['0.3', '0.1'],col_dict):
            Variance = np.linspace(0.01, 0.18, 48)
            spl_1 = np.polyfit(table_v[sim_name], table_k[sim_name],2)
            table_k[sim_name] = np.poly1d(spl_1)(Variance)
            spl_2 = np.polyfit(table_v[sim_name], table_s[sim_name], 2)
            table_s[sim_name] =  np.poly1d(spl_2)(Variance)

            #
            #print(table_s[sim_name])
            plt.plot(Variance, np.array(table_s[sim_name]), label='Storage (eta = ' + linename + ")", color=col)
            plt.plot(Variance, np.array(table_k[sim_name]), label='Generation (eta = ' + linename + ")" , color=col, linestyle = "--")
            plt.xlabel('Variance of supply')
            plt.legend()
            plt.ylabel('Capital investment (Gw)')
            print(table_s[sim_name])
        
        plt.savefig("Results/{}/storage_gen_var.png".format(model_name))

        plt.close()
        for sim_name, linename,col in zip(['array_1', 'array_2'], ['0.3','0.1'],col_dict):
            spl_3 = np.polyfit(table_v[sim_name], np.squeeze(table_u[sim_name]), 2)
            table_u[sim_name] =  np.poly1d(spl_3)(Variance)      #
            plt.plot(table_v[sim_name], np.array(table_u[sim_name]), label='Storage (eta = ' + linename + ")", color=col)
            plt.xlabel('Variance of supply')
            plt.legend()
            plt.ylabel('Weflare')
            print(table_s[sim_name])

        plt.savefig("Results/{}/storage_u_var.png".format(model_name))


        plt.close()
        for sim_name, linename,col in zip(['array_1', 'array_2'], ['0.3', '0.1'],col_dict):
        
           # spl_1 = np.polyfit(table_rs[sim_name], table_k[sim_name],1)
            #table_k[sim_name] = np.poly1d(spl_1)(table_rs[sim_name])
            #spl_2 = np.polyfit(table_rs[sim_name], table_s[sim_name],1)
            #table_s[sim_name] = np.poly1d(spl_2)(table_rs[sim_name])
            plt.plot(table_rs[sim_name], np.array(table_s[sim_name]), label='Welfare (eta = ' + linename + ")", color=col)
            plt.plot(table_rs[sim_name], np.array(table_k[sim_name]), label='Welfare (eta = ' + linename + ")" , color=col, linestyle = "--")
            plt.xlabel('Expected surplus')
            plt.legend()
            plt.ylabel('Capital investment (Gw)')
            
        
        plt.savefig("Results/{}/storage_gen_rs.png".format(model_name))
        

        # Welfare on variance
        #plt.close()
        #plt.plot(table_v, table_u, label='Welfare', color='red')
        #plt.xlabel('Variance of supply')
        #plt.legend()
        #plt.ylabel('Welfare')
        #plt.savefig("Results/{}/{}/storage_u_var_{}.png".format(model_name, sim_name,sim_name))

        # Capital on storage price
        #plt.close()
        #plt.plot(table_rs, table_s, label='Storage', color='red')
        #plt.plot(table_rs, table_k, label='Generation', color='red')
        #plt.xlabel('Storage capital price')
        #plt.legend()
        #plt.ylabel('Capital investment')
        #plt.savefig("Results/{}/{}/storage_gen_rs_{}.png".format(model_name, sim_name,sim_name))

        # Welfare on storage price
        #plt.close()
        #plt.plot(table_rs, table_u, label='Welfare', color='red')
       #plt.xlabel('Storage capital price')
        #plt.legend()
       # plt.ylabel('Welfare')
        #plt.savefig("Results/{}/{}/storage_u_rs_{}.png".format(model_name, sim_name,sim_name))
    """
    """
    """

    """
    if comm.rank == 0:
    
        from scipy import stats
        from matplotlib.ticker import FuncFormatter

        def format_tick_labels(x):

            y = []
            for i in range(len(x)):
                float_str = '{0:.3g}'.format(x[i])
                if "e" in float_str:
                    base, exponent = float_str.split("e")
                    float_str =  r"{0} $\times 10^{{{1}}}$".format(base, int(exponent))
                else:
                    pass 
                y.append(float_str)
            return y

        swe_price =  np.genfromtxt('Settings/ERCOT/price_swe.csv', delimiter=',')*1e3
        

        sim_name = 'baseline_1'

        results_static_og = pickle.load(open("/scratch/tp66/{}/baseline_nostor/baseline_nostor_3_endog.mod"\
                    .format(model_name),\
                            "rb"))
        results_static = runres(results_static_og,model_name, 'baseline_nostor', 'baseline_nostor_3_endog', 1, 4, plot= False)

        swe_price[np.where(swe_price<=0)] = 0.001
        density = stats.kde.gaussian_kde(np.log(swe_price)[~np.isnan(np.log(swe_price))])
        density.covariance_factor = lambda : .4
        density._compute_covariance()
        x = np.linspace(-10,30,100)
        plt.close()
        figure, axes = plt.subplots(nrows=1, ncols=1)
        axes.plot(x, density(x), label = 'Today', color = np.array([227, 27, 35])/255)
        axes.fill_between(x, 0, density(x), alpha = .8, color = np.array([227, 27, 35])/255)
        
        density = stats.kde.gaussian_kde(np.log(results_static['price'])[~np.isnan(np.log(results_static['price']))])
        density.covariance_factor = lambda : .4
        density._compute_covariance()
        axes.plot(x, density(x), label = 'No storage', color = np.array([0, 45, 106])/255)
        axes.fill_between(x, 0, density(x), alpha = .8, color = np.array([0, 45, 106])/255)

        density = stats.kde.gaussian_kde(np.log(results[sim_name][0]['price'])[~np.isnan(np.log(results['baseline_1'][0]['price']))])
        density.covariance_factor = lambda : .4
        density._compute_covariance()
        axes.plot(x, density(x), label = 'Baseline', color = np.array([255, 195, 37])/255)
        axes.fill_between(x, 0, density(x), alpha = .5, color = np.array([255, 195, 37])/255)
        
        #density = stats.kde.gaussian_kde(np.log(results[sim_name][0]['price'])[~np.isnan(np.log(results['baseline_05'][0]['price']))])
        #density.covariance_factor = lambda : .4
        #density._compute_covariance()
        #axes[1].plot(x, density(x), label = 'Baseline', color = np.array([255, 195, 37])/255)
        #axes[1].fill_between(x, 0, density(x), alpha = .5, color = np.array([255, 195, 37])/255)
        
        #density = stats.kde.gaussian_kde(np.log(results[sim_name][1]['price'])[~np.isnan(np.log(results['baseline_05'][1]['price']))])
        #density.covariance_factor = lambda : .4
        #density._compute_covariance()
        #axes[1].plot(x, density(x), label = 'High var.', color = np.array([0, 45, 106])/255)
        #axes[1].fill_between(x, 0, density(x), alpha = .5, color = np.array([0, 45, 106])/255)

        #density = stats.kde.gaussian_kde(np.log(results[sim_name][2]['price'])[~np.isnan(np.log(results['baseline_05'][1]['price']))])
        #density.covariance_factor = lambda : .4
        #density._compute_covariance()
        #axes[1].plot(x, density(x), label = 'Low var.', color = np.array([227, 27, 35])/255)
        #axes[1].fill_between(x, 0, density(x), alpha = .5, color = np.array([227, 27, 35])/255)
        

        #axes[1].set_xticks([np.log(1), np.log(32000), np.log(10000000), np.log(1e13)])
        axes.set_xticks([np.log(1), np.log(32000), np.log(10000000), np.log(1e13)])
        axes.legend()
        #axes[1].legend()
        axes.set_xlabel('Price $/MWh')
        #axes[0].set_xscale('log')
        axes.set_ylabel('Density')
        #axes[1].set_xlabel('Price $/MWh')
        #axes[1].set_ylabel('Density')
        #axes[1].set_xscale('log')

        ticks = axes.get_xticks()
        
        ticks = np.exp(ticks)*1e-3
        print(format_tick_labels(ticks))

        axes.set_xticklabels(format_tick_labels(ticks))
        #axes[1].set_xticklabels(format_tick_labels(ticks))
        #axes[0].xaxis.set_major_formatter(FuncFormatter(format_tick_labels))
       #axes[1].xaxis.set_major_formatter(FuncFormatter(format_tick_labels))


        figure.tight_layout()

        plt.savefig("Results/{}/price_kde_{}_main.png".format(model_name, sim_name))


        # Save just the portion _inside_ the second axis's boundaries
        #extent = axes.get_window_extent().transformed(figure.dpi_scale_trans.inverted())
        #figure.savefig("Results/{}/price_kde_{}_base.png".format(model_name, sim_name), bbox_inches=extent.expanded(1.1, 1.2))

        # Pad the saved area by 10% in the x-direction and 20% in the y-direction
        #fig.savefig('ax2_figure_expanded.png', bbox_inches=extent.expanded(1.1, 1.2))
        
        # Plot demand densities 
    """
    if comm.rank == 0:
        from scipy import stats
        from matplotlib.ticker import FuncFormatter

        def format_tick_labels(x):


            y = []
            for i in range(len(x)):
                float_str = '{0:.3g}'.format(x[i])
                if "e" in float_str:
                    base, exponent = float_str.split("e")
                    float_str =  r"{0} $\times 10^{{{1}}}$".format(base, int(exponent))
                else:
                    pass 
                y.append(float_str)
            return y

        cons =  np.genfromtxt('Settings/ERCOT/demand_erc.csv', delimiter=',')*1e-3
        
        for sim_name in ['baseline_3','baseline_05','baseline_1' ]:

            #@sim_name = 'baseline_3'

            results_static_og = pickle.load(open("/scratch/tp66/{}/baseline_nostor/baseline_nostor_1_endog.mod"\
                        .format(model_name),\
                                "rb"))
            results_static = runres(results_static_og,model_name, 'baseline_nostor', 'baseline_nostor_1_endog', 1, 4, plot= False)
            cons = cons[~np.isnan(cons)]
            cons[np.where(cons<=0)] = 0.001
            density = stats.kde.gaussian_kde(np.log(cons)[~np.isnan(np.log(cons))])
            density.covariance_factor = lambda : .4
            density._compute_covariance()
            x = np.linspace(0,10,100)
            plt.close()
            figure, axes = plt.subplots(nrows=1, ncols=1)
            axes.plot(x, density(x), label = 'Today', color = np.array([227, 27, 35])/255)
            axes.fill_between(x, 0, density(x), alpha = .8, color = np.array([227, 27, 35])/255)
            
            density = stats.kde.gaussian_kde(np.log(results_static['demand'])[~np.isnan(np.log(results_static['demand']))])
            density.covariance_factor = lambda : .4
            density._compute_covariance()
            axes.plot(x, density(x), label = 'No storage', color = np.array([0, 45, 106])/255)
            axes.fill_between(x, 0, density(x), alpha = .8, color = np.array([0, 45, 106])/255)

            density = stats.kde.gaussian_kde(np.log(results[sim_name][0]['demand'])[~np.isnan(np.log(results[sim_name][0]['demand']))])
            density.covariance_factor = lambda : .4
            density._compute_covariance()
            axes.plot(x, density(x), label = 'Baseline', color = np.array([255, 195, 37])/255)
            axes.fill_between(x, 0, density(x), alpha = .5, color = np.array([255, 195, 37])/255)
            
            #density = stats.kde.gaussian_kde(np.log(results[sim_name][0]['price'])[~np.isnan(np.log(results['baseline_05'][0]['price']))])
            #density.covariance_factor = lambda : .4
            #density._compute_covariance()
            #axes[1].plot(x, density(x), label = 'Baseline', color = np.array([255, 195, 37])/255)
            #axes[1].fill_between(x, 0, density(x), alpha = .5, color = np.array([255, 195, 37])/255)
            
            #density = stats.kde.gaussian_kde(np.log(results[sim_name][1]['price'])[~np.isnan(np.log(results['baseline_05'][1]['price']))])
            #density.covariance_factor = lambda : .4
            #density._compute_covariance()
            #axes[1].plot(x, density(x), label = 'High var.', color = np.array([0, 45, 106])/255)
            #axes[1].fill_between(x, 0, density(x), alpha = .5, color = np.array([0, 45, 106])/255)

            #density = stats.kde.gaussian_kde(np.log(results[sim_name][2]['price'])[~np.isnan(np.log(results['baseline_05'][1]['price']))])
            #density.covariance_factor = lambda : .4
            #density._compute_covariance()
            #axes[1].plot(x, density(x), label = 'Low var.', color = np.array([227, 27, 35])/255)
            #axes[1].fill_between(x, 0, density(x), alpha = .5, color = np.array([227, 27, 35])/255)
            

            #axes[1].set_xticks([np.log(1), np.log(32000), np.log(10000000), np.log(1e13)])
            #axes.set_xticks([np.log(1), np.log(32000), np.log(10000000), np.log(1e13)])
            axes.legend()
            #axes[1].legend()
            axes.set_xlabel('Demand (Gwh)')
            #axes[0].set_xscale('log')
            axes.set_ylabel('Density')
            #axes[1].set_xlabel('Demand (Gwh)')
            #axes[1].set_ylabel('Density')
            #axes[1].set_xscale('log')

            ticks = axes.get_xticks()
            
            ticks = np.exp(ticks)
            #print(format_tick_labels(ticks))

            axes.set_xticklabels(format_tick_labels(ticks))
           #axes[1].set_xticklabels(format_tick_labels(ticks))
            #axes[0].xaxis.set_major_formatter(FuncFormatter(format_tick_labels))
            #axes[1].xaxis.set_major_formatter(FuncFormatter(format_tick_labels))


            figure.tight_layout()

            plt.savefig("Results/{}/cons_kde_{}_main.png".format(model_name, sim_name))

            
                
