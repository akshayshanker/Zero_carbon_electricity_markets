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


def tabulate_storage(modlist, row_names, filename, model_name, sim_name):

    results = {}
    table = []
    table_rs = []
    table_u = []
    table_k = []
    table_s = []
    table_v = []

    for key, rname in zip(modlist, row_names):

        # Unpack results, make table
        print("Calculating results from {}".format(key))
        og = pickle.load(
            open("/scratch/kq62/main_v_2/{}/{}.mod".format(sim_name, key), "rb"))
        results[key] = runres(model_name, sim_name, key, 1, 4, plot= False)
        results_row = [
            rname,
            "%.2f (%.2f)" %
            (results[key]['mean_generation'],
             results[key]['var_generation']),
            "%.2f" %
            results[key]['K'],
            "%.2f" %
            results[key]["S_bar_star"],
            "%.2f (%.2f)" %
            (results[key]['mean_price'] *
             1e-3,
             results[key]['var_price'] *
             1e-3),
            "%.2f (%.2f)" %
            (results[key]['mean_demand'],
             results[key]['var_demand']),
            "%.2f (%.2f)" %
            (results[key]['mean_stor'],
             results[key]['var_stor']),
            "%.2f " %
            results[key]['WF']]

        table.append(results_row)
        table_rs.append(og.r_s)
        table_u.append(results[key]['WF'])
        table_k.append(og.K)
        table_s.append(og.S_bar_star)
        table_v.append(og.s_supply)

    header = [
        "Av gen.",
        "Gen. cap.",
        "S cap.",
        "Pr.",
        "Dem.",
        "Av str.",
        "Welfare"]  # , "lowstor %"]
    restab = open(
        "Results/{}/results_tab.tex".format(model_name + '/' + sim_name + '/'), 'w')
    restab.write(
        tabulate(
            table,
            headers=header,
            tablefmt="latex_booktabs",
            floatfmt=".2f"))
    restab.close()

    return table_rs, table_u, table_k, table_s, table_v


if __name__ == '__main__':

    array = genfromtxt('Settings/baseline_1.csv', delimiter=',')[1:]

    model_name = 'main_v_2'
    sim_name = 'array_2_rs'
    settings_file = 'array_2_rs'

    array = genfromtxt('Settings/{}.csv'
                       .format(settings_file), delimiter=',')[1:]

    modlist = []
    row_names = []

    for i in range(len(array)):
        modlist.append('{}_{}_endog'.format(sim_name, i))
        row_names.append(array[i, -1])

    table_rs, table_u, table_k, table_s, table_v = tabulate_storage(
        modlist, row_names, 'baselines', model_name, sim_name)

    # Capital on variance
    plt.close()
    plt.plot(table_v, table_s, label='Storage', color='red')
    plt.plot(table_v, table_k, label='Generation', color='red')
    plt.xlabel('Variance of supply')
    plt.legend()
    plt.ylabel('Capital investment')
    plt.savefig("Results/{}/storage_gen_var_{}.png".format(model_name, sim_name))

    # Welfare on variance
    plt.close()
    plt.plot(table_v, table_u, label='Welfare', color='red')
    plt.xlabel('Variance of supply')
    plt.legend()
    plt.ylabel('Welfare')
    plt.savefig("Results/{}/storage_u_var_{}.png".format(model_name, sim_name))

    # Capital on storage price
    plt.close()
    plt.plot(table_rs, table_s, label='Storage', color='red')
    plt.plot(table_rs, table_k, label='Generation', color='red')
    plt.xlabel('Gen. cap. price ')
    plt.legend()
    plt.ylabel('Capital investment')
    plt.savefig("Results/{}/storage_u_rs_{}.png".format(model_name, sim_name))

    # Welfare on storage price
    plt.close()
    plt.plot(table_rs, table_u, label='Welfare', color='red')
    plt.xlabel('Gen. cap. price')
    plt.legend()
    plt.ylabel('Welfare')
    plt.savefig("Results/{}/storage_u_rs_{}.png".format(model_name, sim_name))


""""

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
