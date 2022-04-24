import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t,spearmanr
import scipy.optimize as optimize

from risk_management import psd


class PriceSimulator():
    """
    A Brownian motion class constructor
    """

    def __init__(self, S=100, mu=0, sigma=0.01):
        "path: # of simulation paths"
        self.S = S
        self.mu = mu
        self.sigma = sigma

    def simulate(self, method="Brownian", path=0, known_step=100, nOfDarws=1000):

        # calculate the each path in Brownian method
        def BrownianPath(start, N):
            """
            Models a stock price S(t) using the Classical Brownian Motion as
            S(t) = S(t-1) + rt
            rt ~ N(0, 𝜎2)
            """
            positions = np.zeros(N)
            for y in range(N):
                positions[y] = start
                start = start + np.random.normal(self.mu, self.sigma)

            return positions

        # calculate the each path in  method
        def ArithmeticPath(start, N):
            """
            Models a stock price S(t) using the Arithmetic Return System as
            S(t) = S(t-1) (1 + rt)
            rt ~ N(0, 𝜎2)
            """
            positions = np.zeros(N)
            for y in range(N):
                positions[y] = start
                start = start * (1 + np.random.normal(self.mu, self.sigma))

            return positions

        # calculate the each path in Geo Brownian method
        def GeoBrownianPath(start, N):
            """
            Models a stock price S(t) using the Geometric Brownian Motion as
            S(t) = S(t-1)*e^rt
            rt ~ N(0, 𝜎2)
            """
            positions = np.zeros(N)
            for y in range(N):
                positions[y] = start
                start = start * np.exp(np.random.normal(self.mu, self.sigma))

            return positions

        """
        Arguments:
            method: Brownian, Arithmetic, GeoBrownian

        Returns:
            simulation_bm: DataFrame of stock prices simulations
        """
        simulation_bm = pd.DataFrame()
        if method == "Brownian":
            Ft_routine = BrownianPath(self.S, known_step)
        elif method == "Arithmetic":
            Ft_routine = ArithmeticPath(self.S, known_step)
        elif method == "GeoBrownian":
            Ft_routine = GeoBrownianPath(self.S, known_step)
        else:
            raise ValueError("no such method!")

        if len(Ft_routine) == 0:
            k_start = self.S
        else:
            k_start = Ft_routine[-1]

        for x in range(path):
            if method == "Brownian":
                Fs_routine = BrownianPath(k_start, nOfDarws)
            elif method == "Arithmetic":
                Fs_routine = ArithmeticPath(k_start, nOfDarws)
            elif method == "GeoBrownian":
                Fs_routine = GeoBrownianPath(k_start, nOfDarws)

            allPath = np.hstack((Ft_routine, Fs_routine))
            simulation_bm[x] = allPath
            self.simulation_bm = simulation_bm
        return simulation_bm


def Return_norm_simulation(array, size):
    mu = array.mean()
    sig = array.std()
    simu_norm = norm.rvs(loc= mu,scale = sig,size = size)
    return simu_norm

def Return_T_params(array):
    # T Distribution by MLE fit
    def t_log_lik(par_vec, x):
        lik = -np.log(t(df=par_vec[0], loc=par_vec[1], scale=par_vec[2]).pdf(x)).sum()
        return lik

    cons = ({'type': 'ineq', 'fun': lambda x: x[0] - 2},
            {'type': 'ineq', 'fun': lambda x: x[2]})

    df, mean, scale = optimize.minimize(fun=t_log_lik,
                                        x0=[2, np.array(array).mean(), np.array(array).std()],
                                        constraints=cons,
                                        args=(np.array(array))).x

    return df, mean, scale

def plot_simulation(simulator):
    data = simulator.data
    fig, axes = plt.subplots(figsize=(10, 8))
    axes.plot(data)
    axes.set_title(f"{simulator.method}, {simulator.n_step} Ways")
    if simulator.known_step!=0:
        axes.axvline(x=simulator.known_step, ls="--", c="r")
    plt.show()


def copula_t_simulation(ret, nOfDarws):
    '''
    Params: ret should be the rate of return
    nOfDarws: # simulations
    '''
    n = ret.shape[1]
    stock_cdf = pd.DataFrame()
    t_params = []

    for col in ret.columns:
        ret[col] -= ret[col].mean()
        df, mean, scale = Return_T_params(ret[col])
        t_params.append([df, mean, scale])
        stock_cdf[col] = t.cdf(ret[col], df=df, loc=mean, scale=scale)

    Corr_spearman = spearmanr(stock_cdf)[0]

    cholesky = psd.chol_psd(Corr_spearman)
    simuNormal = pd.DataFrame(norm.rvs(size=(n, nOfDarws)))
    simulatedT = (cholesky @ simuNormal).T

    Simu_data = pd.DataFrame()
    for i in range(n):
        simu = norm.cdf(simulatedT.iloc[:, i])
        Simu_data[ret.columns[i]] = t.ppf(simu, df=t_params[i][0], loc=t_params[i][1], scale=t_params[i][2])

    return Simu_data


from numpy import *


def simulate_path(s0, mu, sigma, horizon, timesteps, n_sims):
    # Read parameters
    S0 = s0  # initial spot level
    r = mu  # mu = rf in risk neutral framework
    T = horizon  # time horizion
    t = timesteps  # number of time steps
    n = n_sims  # number of simulation

    # Define dt
    dt = T / t  # length of time interval

    # Simulating 'n' asset price paths with 't' timesteps
    S = zeros((t, n))
    S[0] = S0

    for i in range(0, t - 1):
        w = random.standard_normal(n)  # psuedo random numbers
        S[i + 1] = S[i] * (1 + r * dt + sigma * sqrt(dt) * w)  # vectorized operation per timesteps
        # S[i+1] = S[i] * exp((r - 0.5 * sigma ** 2) * dt + sigma * sqrt(dt) * w)            # alternate form

    return S


if __name__=="__main__":
    data_sumu = PriceSimulation(100,known_step=0,method='GeoBrownian',sigma=0.1)
    plot_simulation(data_sumu)
    a = Return_norm_simulation(data_sumu.data, size=100)
    print(a)
