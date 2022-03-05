import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t
import scipy.optimize as optimize

class PriceSimulation():
    """
    A Brownian motion class constructor
    """

    def __init__(self, path=1, method="Brownian", s0=100, known_step=100, n_step=1000, mu=0, sigma=0.01):
        "path: # of simulation paths"
        self.path = path
        self.method = method
        self.s0 = s0
        self.known_step = known_step
        self.n_step = n_step
        self.mu = mu
        self.sigma = sigma
        self.data = self.Simulation(method=method)

    def Simulation(self, method="Brownian"):

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
            Ft_routine = BrownianPath(self.s0, self.known_step)
        elif method == "Arithmetic":
            Ft_routine = ArithmeticPath(self.s0, self.known_step)
        elif method == "GeoBrownian":
            Ft_routine = GeoBrownianPath(self.s0, self.known_step)
        else:
            raise ValueError("no such method!")
        if len(Ft_routine)==0:
            k_start = self.s0
        else:
            k_start = Ft_routine[-1]

        for x in range(self.path):
            if method == "Brownian":
                Fs_routine = BrownianPath(k_start, self.n_step)
            elif method == "Arithmetic":
                Fs_routine = ArithmeticPath(k_start, self.n_step)
            elif method == "GeoBrownian":
                Fs_routine = GeoBrownianPath(k_start, self.n_step)

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
    
if __name__=="__main__":
    data_sumu = PriceSimulation(100,known_step=0,method='GeoBrownian',sigma=0.1)
    plot_simulation(data_sumu)
    a = Return_norm_simulation(data_sumu.data, size=100)
    print(a)
