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
from sklearn.utils.extmath import cartesian

def latex_float(f):
    float_str = "{0:.2g}".format(np.float64(f))
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} $\times 10^{{{1}}}$".format(base, int(exponent))
    else:
        return float_str

def tabulate_storage(V_index,\
                        K_index,\
                        S_index,\
                        filename,\
                        model_name,\
                        sim_name,\
                        tab = False):
    #results = {}
    table = []
    table_rs = np.empty((6,48))
    table_ds = np.empty((6,48))
    table_u = np.empty((6,48))
    table_k = np.empty((6,48))
    table_s = np.empty((6,48))
    table_v = np.empty((6,48))
    table_pi_s = np.empty((6,48))
    table_pi_k = np.empty((6,48))
    VS_range = np.linspace(.01, .18, 4)

    # Load no storage results in table 
    results = pickle.load(open("/scratch/pv33/{}/static/{}/model_nostor_static_res.pkl"\
                    .format(model_name,sim_name),\
                            "rb"))
    results['r_s_star'] = results['r_s_star']*1e-9
    results['r_k_star'] = results['r_k_star']*1e-9

    if tab == True:
            results_row = [
                "%.2f" %
                results['K'],
                "%.2f" %
                results["S_bar_star"],
                "%.2f (%.2f)" % (results['mean_generation'],
                 results['var_generation']),
                latex_float(results['mean_price']*
                 1e-6 ) +
                 "(" + latex_float(results['var_price']*
                 1e-6 ) + ")",
                "%.2f (%.2f)" %
                (results['mean_demand'],
                 results['var_demand']),
                "%.2f (%.2f)" %
                (results['mean_stor'],
                 results['var_stor']),
                latex_float(results['r_s_star']),
                latex_float(results['r_k_star']),
                ]
            table.append(results_row)

    # First plot for different vals of K on S
    for i_v in V_index:
        for i_k in K_index:
            for i_s in S_index:

                # Unpack results, make table
                print("Calculating results from {}".format(i_k))

                results = pickle.load(
                    open("/scratch/pv33/{}/static/{}/model_{}_{}_{}_static_res.pkl"\
                            .format(model_name,sim_name,\
                                        i_s,\
                                        i_k,\
                                        i_v),\
                                        "rb"))

                results['r_s_star'] = results['r_s_star']*1e-9
                results['r_k_star'] = results['r_k_star']*1e-9

                d = results['demand']
                rang = np.arange(1e7-1)

                if tab == True:
                    print(results['var_generation'])

                    results_row = [
                    "%.2f (%.2f)" % (results['mean_generation'],
                        results['var_generation']),
                        "%.2f" %
                        results['K'],
                        "%.2f" %
                        results["S_bar_star"],
                        latex_float(results['mean_price']*1e-6 ) +
                         "(" + latex_float(results['var_price']*1e-6) + ")",
                        "%.2f (%.2f)" %
                        (results['mean_demand'],
                         results['var_demand']),
                        "%.2f (%.2f)" %
                        (results['mean_stor'],
                         results['var_stor']),
                        latex_float(results['r_s_star']),
                        latex_float(results['r_k_star']),
                        ]
                    
                    table.append(results_row)
                    
                table_ds[i_v,np.where(K_index == i_k)] = results['var_demand']
                table_rs[i_v,np.where(K_index == i_k)] = results['var_price']
                table_u[i_v,np.where(K_index == i_k)] = np.mean(d)
                table_k[i_v,np.where(K_index == i_k)] = results['K']
                table_s[i_v,np.where(K_index == i_k)] = results["S_bar_star"]
                table_v[i_v,np.where(K_index == i_k)] = VS_range[i_v]
                table_pi_s[i_v,np.where(K_index == i_k)] = results['r_s_star']
                table_pi_k[i_v,np.where(K_index == i_k)] = results['r_k_star']

    header = [
        "Av gen.",
        "Gen. cap.",
        "Str. cap.",
        "Pr.",
        "Dem.",
        "Av str."
        , "r_k", "r_s"]  # , "lowstor %"]
    restab = open(
        "Results/{}/static_results_tab.tex".format(model_name + '/' + sim_name + '/'), 'w')
    restab.write(
        tabulate(
            table,
            headers=header,
            tablefmt="latex_booktabs",
            floatfmt=".2f"))
    restab.close()
    
    return table_ds,table_rs, table_u, table_k, table_s, table_v, table_pi_s,table_pi_k, table


if __name__ == '__main__':

    array = genfromtxt('Settings/baseline_1.csv', delimiter=',')[1:]
    model_name = 'main_v_2'
    import seaborn as sns


    """
    # First get table

    K_index = np.array([0])
    S_index = np.array([0,10,40, 47])
    V_index = np.array([2])

    #all_tabs = []
    #all_tabs.append('$\eta$ = 0.3')
    table_rs, table_u, table_k, table_s,\
    table_v, table_pi_s,table_pi_k, table\
             = tabulate_storage(V_index,\
                                K_index,\
                                S_index,\
                                'baselines',\
                                model_name,\
                                'baseline_3',\
                                tab= True)

    #all_tabs.append(table)
    #all_tabs.append('$\eta = 0.1')
    table_rs, table_u, table_k, table_s,\
    table_v, table_pi_s,table_pi_k, table_1\
             = tabulate_storage(V_index,\
                                K_index,\
                                S_index,\
                                'baselines',\
                                model_name,\
                                'baseline_1',\
                                tab= True)
    table = table+ table_1
    #all_tabs.append('$\eta = 0.05')
    table_rs, table_u, table_k, table_s,\
    table_v, table_pi_s,table_pi_k, table_05\
             = tabulate_storage(V_index,\
                                K_index,\
                                S_index,\
                                'baselines',\
                                model_name,\
                                'baseline_05',\
                                tab= True)
    table = table + table_05

    header = [
    "Av gen.",
    "Gen. cap.",
    "Str. cap.",
    "Pr.",
    "Dem.",
    "Av str."
    , "r_k", "r_s"]  # , "lowstor %"]
    restab = open(
        "Results/{}/all_static.tex".format(model_name), 'w')
    restab.write(
        tabulate(
            table,
            headers=header,
            tablefmt="latex_raw"))
    restab.close()
    """
    
    # Plot storage returns on S_bar for different values of K
    # Loop over different elasicity of demands 

    for e in('3','1'): 
        cmap_here = sns.diverging_palette(220, 20, as_cmap=True)
    
        S_index = np.array([0])
        K_index = np.arange(48)
        V_index = np.array([0,1,2,3])

        K_range =  np.linspace(30, 100, 48)
        S_range =  np.linspace(30, 100, 48)
        VS_range = np.linspace(.01, .18, 4)

        sim_name ='baseline_' + e

        #all_tabs = []
        #all_tabs.append('$\eta$ = 0.3')
        table_ds,table_rs, table_u, table_k, table_s,\
        table_v, table_pi_s,table_pi_k, table\
                 = tabulate_storage(V_index,\
                                    K_index,\
                                    S_index,\
                                    'baselines',\
                                    model_name,\
                                    'baseline_{}'.format(e),\
                                    tab = False)

        plt.close()
        figure, axes = plt.subplots(nrows=1, ncols=2)
        colors = cmap_here(np.linspace(0,1,6))
        for i_v in range(len(V_index)):
            axes[0].plot(table_k[i_v], table_pi_s[i_v], label='K = {:.2f}'\
                .format(VS_range[V_index[i_v]],e), color=colors[i_v])
        axes[0].set_xlabel('Generation')
        #axes[0].legend()
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        axes[0].set_ylabel('Storage returns')
            
        #plt.savefig("Results/{}/{}/static/returns_r_s_S.png"\
        #    .format(model_name, sim_name))

        plt.close()
        colors = cmap_here(np.linspace(0,1,6))
        for i_v in range(len(V_index)):
            axes[1].plot(table_k[i_v], table_u[i_v], label='K = {:.2f}'\
                .format(VS_range[V_index[i_v]],e), color=colors[i_v])
        axes[1].set_xlabel('Generation')
        axes[1].legend()
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        axes[1].set_ylabel('Utility')
            
        figure.savefig("Temp/util_k_{}.png"\
            .format(e))

        plt.close()
        figure, axes = plt.subplots(nrows=1, ncols=2)
        colors = cmap_here(np.linspace(0,1,6))
        for i_v in range(len(V_index)):
            axes[0].plot(table_k[i_v], table_rs[i_v]*1e-4, label = 'K = {:.2f}'\
                .format(VS_range[K_index[i_v]],e), color = colors[i_v])
        axes[0].set_xlabel('Generation')
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        axes[0].legend()
        axes[0].set_ylabel('Price variance')


        for i_k in range(len(K_index)):
            axes[1].plot(table_s[i_v], table_ds[i_v], label = 'K = {:.2f}'\
            .format(K_range[K_index[i_v]],e), color = colors[i_v])
        axes[1].set_xlabel('Generation')
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        #axes[1].legend()
        axes[1].set_ylabel('Demand variance')
            
        figure.savefig("Temp/variance_{}.png"\
            .format(e))
    
    """


    table2 = [mod3, mod4, mod5]
    print(tabulate(table2, headers = header,  tablefmt="latex_booktabs", floatfmt=".2f"))
    restab = open("results234_tab.tex", 'w')
    restab.write(tabulate(table2, headers = header,  tablefmt="latex_booktabs", floatfmt=".2f"))
    restab.close()


    #table3 = [cmod0, cmod1, cmod2, cmod3, cmod4]

    print(tabulate(table3, headers = header,  tablefmt="latex_booktabs", floatfmt=".2f"))
    restab = open("resultscov_tab.tex", 'w')
    restab.write(tabulate(table3, headers = header,  tablefmt="latex_booktabs", floatfmt=".2f"))
    restab.close()


    swe_price =  np.genfromtxt('price_swe.csv', delimiter=',')

    f, axarr = plt.subplots(1,3, sharey='row')
    axarr[0].boxplot(np.log(swe_price[1:,]))
    axarr[0].set_xlabel('Swedish prices today', fontsize = 10)
    axarr[0].set_ylabel('Log of price')
    axarr[1].boxplot(np.log(resmod0['price']))
    axarr[1].set_xlabel('Current batt.price/ low var.', fontsize = 10)
    axarr[2].boxplot(np.log(resmod3['price']))
    axarr[2].set_xlabel('Low batt. price/ low var', fontsize = 10)
    f.tight_layout()


    plt.savefig("/home/akshay_shanker/model_box_price.png")


    f, axarr = plt.subplots(1,2, sharey='row')
    axarr[0].boxplot(np.log(resmod0['stored']))
    axarr[0].set_xlabel('Current batt. price/ low variance', fontsize = 10)
    axarr[0].set_ylabel('Log of stored power')
    axarr[1].boxplot(np.log(resmod3['stored']))
    axarr[1].set_xlabel('Low batt. price/ low variance', fontsize = 10)
    f.tight_layout()


    plt.savefig("/home/akshay_shanker/model_box_stor.png")

    swe_dem =  np.genfromtxt('dem_swe.csv', delimiter=',')

    f, axarr = plt.subplots(1,3, sharey='row')
    axarr[0].boxplot(np.log(swe_dem[1:,]))
    axarr[0].set_xlabel('Swedish demand today', fontsize = 10)
    axarr[0].set_ylabel('Log of eqm. demand')
    axarr[1].boxplot(np.log(resmod0['demand']))
    axarr[1].set_xlabel('Current batt. price/ low var', fontsize = 10)
    axarr[2].boxplot(np.log(resmod3['demand']))
    axarr[2].set_xlabel('Low batt. price/ low var', fontsize = 10)

    f.tight_layout()


    plt.savefig("/home/akshay_shanker/model_box_dem.png")


    f, axarr = plt.subplots(1,3, sharey='row')
    axarr[0].boxplot(np.log(swe_dem[1:,]))
    axarr[0].set_xlabel('Swedish demand today', fontsize = 10)
    axarr[0].set_ylabel('Log of eqm. demand')
    axarr[1].boxplot(np.log(resmod0['demand']))
    axarr[1].set_xlabel('Current batt. price/ low var', fontsize = 10)
    axarr[2].boxplot(np.log(resmod3['demand']))
    axarr[2].set_xlabel('Low batt. price/ low var', fontsize = 10)

    f.tight_layout()

    plt.clf()
    plt.plot(og.grid, og.p_inv(-og.shock_X[2,1],og.rho_star[2])+ og.K*og.shock_X[2,0])
    plt.plot(og.grid, og.p_inv(-og.shock_X[3,1],og.rho_star[3])+ og.K*og.shock_X[3,0])
    """
