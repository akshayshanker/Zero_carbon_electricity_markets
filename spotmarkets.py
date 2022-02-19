

"""
This module contains the EmarketModel class and functions and operators
requred to solve a electricity spot market model

Classes: EmarketModel
         Class with parameters and function for the electricity
          spot market model

Functions: time_operator_factory
            Generates the operators require to solve an instance of
            electricity spot market model

"""


# import modules interp_as
import numpy as np
from interpolation import interp
from quantecon.optimize.scalar_maximization import brent_max
from scipy.optimize import brentq
from quantecon.optimize.root_finding import brentq as brentq_qe

import matplotlib.pyplot as plt
from itertools import product
from numba import njit, prange, jit

from helpfuncs.fixedpoint import fixed_point
from sklearn.utils.extmath import cartesian
from scipy.optimize import broyden1
from scipy.stats import truncnorm
from scipy.optimize import fsolve
from helpfuncs import config
import time
from scipy import stats
from helpfuncs.helperfuncs import interp_as


from interpolation import interp
from interpolation.splines import UCGrid, CGrid, nodes, eval_linear


class EmarketModel:
    """
    This class takes parameters and functions from a sport market
    problem and creates a EmarketModel class

    An instance of this class is a  model with parameters,
    functions, asset grids and shock processes.

    The Emarket model can then be solved using operators generated by
    time_operator_factory

    Parameters
    ----------
    beta : float
           Hourly discount rate
    delta_storage: float
                    Hourly depreciation rate

    zeta_storage:   float
                     pip constraint parameter
    grid_min_s:     float
                     min storage
    mu_supply:      float
                     mean supply shock
    s_supply:       float
                     standard deviation of supply
    mu_demand:      float
                      mean demand
    s_demand:       float
                     standard deviation of supply shock
    grid_size:      int
                     size of storage grid
    grid_size_s:    int
                     size of supply grid
    grid_size_d:    int
                     size of demand grid
    r_s:            float
                     price of storage capital

    r_k:            float
                     price of generation capital
    D_bar           float
                     demand curve shifter
    TS_length       int
                     length of timeseries to estimate stat distributions
    eta_demand      float
                     elasticity of demand
    """

    def __init__(self,
                 beta=.91**(1 / (365 * 24)),
                 delta_storage=.005,
                 zeta_storage=.1,
                 grid_min_s=1e-100,
                 mu_supply=0.291,
                 s_supply=0.0286,
                 grid_max_x=100,
                 mu_demand=.5,
                 s_demand=.15,
                 grid_size=100,
                 grid_size_s=3,
                 grid_size_d=3,
                 r_s=.01,
                 r_k=2,
                 D_bar=10,
                 sol_func=np.array(0),
                 TS_length=1E6,
                 eta_demand=.5
                 ):

        # model parameters
        self.beta, self.delta_storage, self.zeta_storage = beta, delta_storage, zeta_storage

        self.D_bar = D_bar

        self.eta_demand = eta_demand

        # length of TS to generate first stage profits
        self.TS_length = TS_length

        #   capital rates of return
        self.r_s, self.r_k = r_s, r_k

        self.D_bar = D_bar

        # model functions
        @njit
        def pr(e, x, D_bar):
            """
            Price function. Or inverse of demand function.

            Parameters
            ----------
            e:      float
                     demand shock value
            x:      float
                     level of demand
            D_bar:  float
                     demand shifter
            Returns
            -------
            price:  float
                     price level
            """

            if x > 1e-350:

                return (D_bar**(1 / eta_demand))\
                    * np.exp(e / eta_demand) * x**(-1 / eta_demand)
            else:
                return np.inf

        @njit
        def p_inv(e, x, D_bar):
            """
            Demand function.

            Parameters
            ----------
            e:      float
                     demand shock value
            x:      float
                     price
            D_bar:  float
                     demand shifter
            Returns
            -------
            demand:  float
            """

            return D_bar * np.exp(e) * (np.power(x, (-eta_demand)))

        self.pr = pr
        self.p_inv = p_inv

        # initialize grid sizes  and grid

        self.grid_size, self.grid_min_s = grid_size, grid_min_s

        self.grid_size_s, self.grid_size_d = grid_size_s, grid_size_d

        self.grid = np.linspace(grid_min_s, grid_max_x, grid_size)
        # alpha and beta parameters for beta distribution of supply

        self.mu_supply, self.s_supply, self.mu_demand, self.mu_supply\
            = mu_supply, s_supply, mu_demand, mu_supply

        self.alpha = mu_supply * \
            (((mu_supply * (1 - mu_supply)) / s_supply) - 1)

        self.iota = (1 - mu_supply) * \
            (((mu_supply * (1 - mu_supply)) / s_supply) - 1)

        self.alpha2 = mu_demand * \
            (((mu_demand * (1 - mu_demand)) / s_demand) - 1)
        self.iota2 = (1 - mu_demand) * \
            (((mu_demand * (1 - mu_demand)) / s_demand) - 1)

        # generate PMFs for supply and demand shocks

        self.demand_shocks = np.genfromtxt(
            'Settings/errors_demand.csv', delimiter=',')

        X_supply_prime = np.linspace(0, 1, self.grid_size_s + 1)
        self.P_supply = np.diff(
            stats.beta.cdf(
                X_supply_prime,
                self.alpha,
                self.iota))
        self.X_supply = (X_supply_prime[1:] + X_supply_prime[:-1]) / 2

        # normalise the mean of the supply shocks so it remains constant
        mean_prime = np.inner(self.X_supply, self.P_supply)
        self.X_supply = self.X_supply * (mu_supply / mean_prime)

        self.P_dem, self.X_dem = np.histogram(
            self.demand_shocks, bins=grid_size_d)
        self.P_dem = self.P_dem / len(self.demand_shocks)

        # generate cartesian products of shocks

        self.shock_X = cartesian([self.X_supply[0:grid_size_s],
                                  self.X_dem[0:grid_size_d]])

        P_tmp = cartesian([self.P_supply, self.P_dem])
        self.P = np.zeros(len(self.shock_X))

        for i in range(len(self.P)):
            self.P[i] = P_tmp[i][0] * P_tmp[i][1]
        mean = np.inner(self.X_supply, self.P_supply)
        variance = np.inner((self.X_supply - mean)**2, self.P_supply)

        print("initialized model class")
        print(
            "Supply shock mean is {} and standard deviation is {}".format(
                mean, variance))


def time_operator_factory(og, parallel_flag=True):
    """A function factory for building the first stage and second stage
    spot market problem operator

    Parameters
    ----------
    og : SpotMarketModel
         Instance of spot market model

    Returns
    -------
    F_star: function
        F* in paper
        function whose fixed point is the equilibrium first stage
        generation capital

    TC_star: function
        function returns second-stage rho_star (optimal pricing function)
        by iterating on T for given K and S

    G: function
        Returns first stage storage profits
        for vals of K and S
    F: function
        returns first stage generator profits
        for vals of K and S
    """

    beta, delta_storage, zeta_storage = og.beta, og.delta_storage,\
                                         og.zeta_storage
    pr, p_inv = og.pr, og.p_inv
    grid_size, grid_min_s, grid_size_s, grid_size_d = og.grid_size, \
                                                        og.grid_min_s,\
                                                        og.grid_size_s,\
                                                        og.grid_size_d
    D_bar = og.D_bar

    P, P_supply, P_dem = og.P, og.P_supply, og.P_dem
    X_dem, X_supply, shock_X = og.X_dem, og.X_supply, og.shock_X

    alpha, iota, alpha2, iota2 = og.alpha, og.iota, og.alpha2, og.iota2
    r_s, r_k = og.r_s, og.r_k

    TS_length = int(og.TS_length)

    D_bar = og.D_bar

    eta_demand = og.eta_demand

    # supply and shocks for simulation
    shock_index = np.arange(len(shock_X))
    shocks = np.random.choice(shock_index, int(og.TS_length), p=P)

    @njit
    def objective_price(c,
                        rho,
                        y,
                        K,
                        S_bar,
                        grid,
                        ):
        """"
        The right hand side of the pricing operator

        Parameters
        ----------
        c :     float
                 value of price today at which operator operator evaluated at
        rho:    array
                 pricing function tomorrow
        y:      array
                 state t as 2-tuple: y[0] gives grid index, y[1] shock index
        K:      float
                 generation capital
        S_bar:  float
                 storage capital

        grid:   numpy array
                 storage grid with S_bar as max value

        Returns
        -------
        RHS: float
                error at state y for pricing operator

        """

        # First turn w into a function via interpolation

        s = grid[y[1]]
        z = shock_X[y[0], 0]
        e = shock_X[y[0], 1]

        B_underbar = max((1 - delta_storage) * s - zeta_storage * S_bar, 0)
        B_upperbar = min(zeta_storage * S_bar + (1 - delta_storage) * s, S_bar)

        # next period storage given price and state today
        s_prime = K * z - p_inv(e, c, D_bar) + (1 - delta_storage) * s

        # generate next period possible state values

        integrand = np.zeros(len(shock_X[:, 0]))

        for i in range(len(shock_X[:, 0])):
            integrand[i] = np.interp(s_prime, grid, rho[i])

        # generate the expected price for the next period

        Eprice = np.dot(P, integrand)

        RHS = min([max([beta * (1 - delta_storage) * Eprice,
                        pr(e,
                           z * K + (1 - delta_storage) * s - B_underbar,
                           D_bar)]),
                   pr(e,
                      z * K + (1 - delta_storage) * s - B_upperbar,
                      D_bar)]) - c
        #print("trying value {} get diff {}".format(c, RHS))
        return RHS

    @njit
    def T(rho, K, S_bar, grid):
        """
        The iteration time operator


        Parameters
        ----------
        rho :   2D numpy array
                 pricing function 
                 (rows index shock index and column index grid index)
        K:      float
                 generation capital
        S_bar:  float
                 storage capital
        grid:   numpy array
                 storage grid with S_bar as max value

        Returns
        -------
        rho_new: float
                error at state y for pricing operator
        """

        grid_index = []

        for index in np.ndenumerate(rho):
            grid_index.append(index[0])

        def brent_func(i):
            if objective_price(1e-300,
                               rho,
                               grid_index[i],
                               K,
                               S_bar,
                               grid) * objective_price(1e300,
                                                       rho,
                                                       grid_index[i],
                                                       K,
                                                       S_bar,
                                                       grid) < 0:
                c_star = brentq_qe(
                    objective_price,
                    1e-300,
                    1e300,
                    disp=False,
                    args=(
                        rho,
                        grid_index[i],
                        K,
                        S_bar,
                        grid))[0]
            else:
                c_star = 1e-100

            return c_star

        rho_new_p = np.zeros(len(grid_index))

        for i in range(len(grid_index)):
            rho_new_p[i] = brent_func(i)

        rho_new = rho_new_p.reshape((len(shock_X), grid_size))

        return rho_new

    def TC_star(v_init, K, S_bar, tol, grid):
        """"
        Fixed point operator
        For initial guess of time operator, returns its fixed point

        Parameters
        ----------
        v_init :    2D numpy array
                     initial pricing function 
                     (rows index shock index and column index grid index)
        K:      float
                 generation capital
        S_bar:  float
                 storage capital
        grid:   numpy array
                 storage grid with S_bar as max value

        Returns
        -------
        rho_new: float
                  fixed point of price function
        """

        rho = v_init
        rho_updated = fixed_point(
            lambda v: T(
                v,
                K,
                S_bar,
                config.grid),
            rho,
            error_flag=0,
            tol=1e-06,
            error_name="pricing")

        return rho_updated

    def G(K, S_bar, tol_TC, tol_pi):
        """
        Function that gives first stage marginal profits,
        or shadow value of
        storage capital for given level of storage capital
        and generation capital minus cost of capital.

        Parameters
        ----------
        K :    float64
                generation capital
        S_bar:  float
                 storage capacity
        tol_TC:  float
                  tolerance for the time iteration fp
        tol_pi:  float
                   number of decimal places to round to when evaluating
                   constraints

        Returns
        ----------
        Pi_hat - r_s:  float
                        net present discounted value of profits

        """

        start = time.time()

        # set up new grid for updated S_bar
        config.grid = np.linspace(grid_min_s, S_bar, grid_size)

        # set up array of grid size for new initial price function
        # this v_init_one is here only a template
        v_init_one = np.ones(config.grid.shape)

        v_init = np.tile(v_init_one, (grid_size_s * grid_size_d, 1))

        # set up interpolant of previous iterations price function
        def v_init_func(
            e,
            s): return np.interp(
            s,
            config.grid_old,
            config.rho_global_old[e])

        # interpolate previous pricing funciton on
        # new grid and set as new initial value
        for i in range(grid_size_s * grid_size_d):
            v_init[i] = v_init_func(i, config.grid)

        # fine the fixed point of the pricing operator
        start_rho = time.time()
        rho_star = TC_star(
            v_init, K, S_bar, tol_TC, np.linspace(
                grid_min_s, S_bar, grid_size))
        end_rho = time.time()

        #print("rho_star took {} to solve".format(end_rho-start_rho))

        config.rho_global_old = rho_star

        config.grid_old = config.grid
        grid = config.grid

        PI_bar = np.zeros(TS_length - 1)

        s = np.zeros(TS_length)
        priceT = np.zeros(TS_length)
        d = np.zeros(TS_length)
        gen = np.zeros(TS_length)
        DST = np.zeros(TS_length)

        #  shocks for simulation
        #shock_index             = np.arange(len(shock_X))
        # Store supply shocks
        #shocks                  = np.random.choice(shock_index, T, p=P)

        s[0] = S_bar / 2

        @njit
        def nextstor(s, d, gen):
            s1 = min([max([grid_min_s, (1 - delta_storage) * s - d + gen]), S_bar])
            s2 = max([min([zeta_storage * S_bar + (1 - delta_storage) * s, s1]
                          ), - zeta_storage * S_bar + (1 - delta_storage) * s])

            return s2

        start_seq = time.time()
        for i in range(TS_length):
            priceT[i] = interp_as(
                grid, rho_star[shocks[i]], np.array([s[i]]))[0]
            d[i] = p_inv(shock_X[shocks[i], 1], priceT[i], D_bar)
            gen[i] = shock_X[shocks[i], 0] * K

            if i < TS_length - 1:
                s[i + 1] = nextstor(s[i], d[i], gen[i])
                DST[i] = (1 - delta_storage) * s[i] - s[i + 1]

        end_seq = time.time()

        #print("Generating serial sequence took {}".format(end_seq - start_seq))

        @njit
        def return_pi(DS, S1, PI_bar):

            if round(DS, tol_pi) >= round(zeta_storage * S_bar, tol_pi):
                return - zeta_storage * PI_bar

            if -round(DS, tol_pi) >= round(zeta_storage * S_bar, tol_pi):
                return zeta_storage * PI_bar
            # if x<0:

            if round(S1, tol_pi) >= round(S_bar, tol_pi):
                return PI_bar

            else:
                return 0

        @njit
        def T2(i):

            integrand = np.zeros(len(shock_X[:, 0]))

            for j in range(len(shock_X[:, 0])):
                integrand[j] = np.interp(s[i + 1], grid, rho_star[j])

            # <- this should have the minmax condition on the RHS
            Eprice = beta * (1 - delta_storage) * np.dot(P, integrand)

            PI_bar = - priceT[i] + Eprice

            PI_hat = return_pi(DST[i], s[i + 1], PI_bar)

            return max([0, PI_hat])

        start_PI = time.time()

        PI_hat = np.zeros(TS_length - 1)

        for i in range(TS_length - 1):
            PI_hat[i] = T2(i)

        err = (1 / (1 - beta)) * np.mean(PI_hat) - r_s
        end = time.time()
        end_PI = time.time()

        return (1 / (1 - beta)) * np.mean(PI_hat) - r_s  # , s, price, gen, d

    def F(K, S_bar, tol):
        """
        Function that gives first stage profits - cost of
        generation capital for given level of storage capital
        and generation capital

        Parameters
        ----------
        K :    float
                generation capital
        S_bar:  float
                 storage capacity
        tol:    float
                  tolerance for the time iteration fp

        Returns
        ----------
        Pi_hat - r_k:  float
                        net present discounted value of profits
        """

        # initial guess of value function is pricing function saved from
        # previous evaluation of G operator
        v_init = config.rho_global_old

        # set-up grid based on value of S_nar
        config.grid = np.linspace(grid_min_s, S_bar, grid_size)

        # calculate pricing function
        #rho_star                = TC_star(v_init,K, S_bar, tol, config.grid)
        #config.rho_global_old   = rho_star
        rho_star = config.rho_global_old

        s = np.zeros(TS_length)
        priceT = np.zeros(TS_length)
        d = np.zeros(TS_length)
        gen = np.zeros(TS_length)
        integrand = np.zeros(TS_length)

        # generate sequence of price and demand
        s[0] = S_bar / 2
        e = np.zeros(TS_length)
        z = np.zeros(TS_length)

        for i in range(TS_length):
            zi = shock_X[shocks[i], 0]
            ei = shock_X[shocks[i], 1]

            priceT[i] = max(0, interp(config.grid, rho_star[shocks[i]], s[i]))
            d[i] = p_inv(ei, priceT[i], D_bar)
            gen[i] = zi * K
            if i < TS_length - 1:
                s[i + 1] = np.max([grid_min_s,
                                   (1 - delta_storage) * s[i] - d[i] + gen[i]])
                s[i + 1] = np.min([s[i + 1], S_bar])

            integrand[i] = priceT[i] * zi

        # integrate to expected price
        Eprice = np.mean(integrand)
        err = (1 / (1 - beta)) * Eprice - r_k

        #print("Error for operator F for generator value %s operator is %s"%(K, err))

        if K <= 1e-1 and (1 / (1 - beta)) * Eprice - r_k < 0:
            return 0
        else:
            return (1 / (1 - beta)) * Eprice - r_k

    def F_star(K, tol_TC, tol_brent_1, tol_brent, tol_pi):
        """
        Function that iterates on the F* operator
        F* operator fices value of eq. gen capital implied
        by eqm. storage capital calculated from arg of F*

        Fixed point to F* is eqm. K


        Parameters
        ----------
        K :         float
                     generation capital
        tol_TC:     float
                     tolerance for time iteration fp
        tol_brent_1: float
                      tolerance for storage capital initial
        tol_brentL   float
                      tolerance for storage capital when refining


        Returns
        ----------
        K_prime:    float
                     iteration of generation capital
        """

        # define function whose zero is the optimal first stage storage capital
        def G_star(S): return G(K, S, tol_TC, tol_pi)

        # calc optimal first stage storage capital given K
        if config.toggle_GF == 0:
            S_star = brentq(G_star, 1e-10, 2000, xtol=tol_brent_1)

        if config.toggle_GF == 1:
            S_star = brentq(
                G_star,
                config.S_bar_star * .75,
                config.S_bar_star * 1.25,
                xtol=tol_brent)  # calc optimal first stage storage capital given K

        # define function whose zero is the optimal first gen storage capital
        # given store cap.
        def F_star(K): return F(K, S_star, tol_TC)
        if config.toggle_GF == 0:
            # calc optimal first stage gen capital given S_bar
            K_star = brentq(F_star, 1e-1, 5000, xtol=tol_brent_1)
        if config.toggle_GF == 1:
            # calc optimal first stage gen capital given S_bar
            K_star = brentq(
                F_star,
                config.K_star * .75,
                config.K_star * 1.25,
                xtol=tol_brent)

        if abs(K - K_star) < 2:
            config.toggle_GF = 0
        if abs(K - K_star) >= 2:
            config.toggle_GF = 0

        config.S_bar_star = S_star
        config.K_star = K_star
        return K_star

    def GF_RMSE(S, K, tol_TC, tol_pi):
        """
        Returns RMSE of first stage profit/loss for storage
        and generation
        capital for given values of S and K. Eqm is where this
        function == zero

        Parameters
        ----------
        K :         float
                     generation capital
        S :         float
                     storage capital
        tol_TC:     float
                     tolerance for time iteration fp
        tol_pi:     float
                     tolerance for number of decimal places to round
                     to when evaluating storage constraints


        """

        stor_pnl = G(K, S, tol_TC, tol_pi)
        gen_pn = F(K, S, tol_TC)

        return np.sqrt(stor_pnl**2 + gen_pn**2)

    return GF_RMSE, F_star, TC_star, G, F
