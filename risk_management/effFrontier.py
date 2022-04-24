
import numpy as np
import pandas as pd
import scipy.optimize as sco
def MinRiskPortfolio(mu: "array-like", cov: "cov matrix", m: "float"):
    """
    @params:
    mu: asset average return, size:n x 1
    simga: asset standard deviation, size: n x 1
    cov: portfolio covariance, size nxn
    m: risk optimization target
    """
    n = len(mu)
    ones = np.ones(n)[:, np.newaxis]

    I = np.linalg.inv(np.array(cov))
    mu = np.array(mu)[:, np.newaxis]

    A = ones.T @ I @ ones
    B = mu.T @ I @ ones
    C = mu.T @ I @ mu
    lamb = (A * m - B) / (A * C - B ** 2)
    gamma = (C - B * m) / (A * C - B ** 2)
    print(1 / (A * C - B ** 2) * I @ ((A * mu - B * ones) * m) + (C * ones - B * mu))
    w = 1 / (A * C - B ** 2) * I @ ((A * mu - B * ones) * m + (C * ones - B * mu))

    risk = np.sqrt(w.T @ cov @ w)[0, 0]
    return w.T, risk


def effFrontier(mu: "array-like", cov: "array-like", scope: "tuple-like", rf: "float", shortValid=True):
    """
    params:
    mu: asset average return, size:n x 1
    cov: portfolio covariance, size nxn
    scope: frontier target range
    rf: risk-free rate
    shortValid: whether short sale is allowed
    """

    def portfolio_stats(weights):
        weights = np.array(weights)[:, np.newaxis]
        port_rets = weights.T @ mu
        port_vols = np.sqrt(weights.T @ cov @ weights)

        return np.array([port_rets, port_vols, (port_rets - rf) / port_vols]).flatten()

    # Maximizing sharpe ratio/ method to find tangency portfolio
    def min_sharpe_ratio(weights):
        return -portfolio_stats(weights)[2]

    # Minimize the volatility
    def min_volatility(weights):
        return portfolio_stats(weights)[1]

    numOfAssets = len(mu)
    cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
    stocks = mu.index
    mu = np.array(mu)
    cov = np.array(cov)
    # print(mu)
    initial_wts = np.array(numOfAssets * [1. / numOfAssets])

    # find efficient frontier
    targetrets = np.linspace(scope[0], scope[1], scope[2])
    tvols = []

    for tr in targetrets:

        # for short sale forbid constraints
        bnds = tuple((0, 1) for x in range(numOfAssets))
        ef_cons = ({'type': 'eq', 'fun': lambda x: portfolio_stats(x)[0] - tr},
                   {'type': 'eq', 'fun': lambda x: sum(x) - 1})

        if shortValid == True:
            opt_ef = sco.minimize(min_volatility, initial_wts, method='SLSQP', constraints=ef_cons)
        else:
            opt_ef = sco.minimize(min_volatility, initial_wts, method='SLSQP', bounds=bnds, constraints=ef_cons)

        tvols.append(opt_ef['fun'])
    targetvols = np.array(tvols)
    # Dataframe for EF
    efPort = pd.DataFrame({
        'targetrets': np.around(targetrets, 4),
        'targetvols': np.around(targetvols, 4),
        'targetsharpe': np.around((targetrets - rf) / targetvols, 4)
    })
    bnds = tuple((0, 1) for x in range(numOfAssets))
    # Tangency Portfolio
    if shortValid == True:
        opt_sharpe = sco.minimize(min_sharpe_ratio, initial_wts, method='SLSQP', constraints=cons)
    else:
        opt_sharpe = sco.minimize(min_sharpe_ratio, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)

    w = opt_sharpe.x
    stat = portfolio_stats(w)
    Tangencyrets, Tangencyvols, Tangencysharpe = stat[0], stat[1], stat[2]

    tangencyPort = pd.Series([Tangencyrets, Tangencyvols, Tangencysharpe], index=['ret', 'vol', 'sharpe'])
    assetWeight = pd.DataFrame(w, index=stocks, columns=['Weight'])
    return efPort, tangencyPort, assetWeight


def tangency_weights(mu: "array-like", cov: "array-like", rf: "float", shortValid=True):
    """
    params:
    mu: asset average return, size:n x 1
    cov: portfolio covariance, size nxn
    rf: risk-free rate
    shortValid: whether short sale is allowed
    """

    def portfolio_stats(weights):
        weights = np.array(weights)[:, np.newaxis]
        port_rets = weights.T @ mu
        port_vols = np.sqrt(weights.T @ cov @ weights)

        return np.array([port_rets, port_vols, (port_rets - rf) / port_vols]).flatten()

    # Maximizing sharpe ratio/ method to find tangency portfolio
    def min_sharpe_ratio(weights):
        return -portfolio_stats(weights)[2]

    numOfAssets = len(mu)
    cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
    stocks = mu.index
    mu = np.array(mu)
    cov = np.array(cov)
    initial_wts = np.array(numOfAssets * [1. / numOfAssets])
    bnds = tuple((0, 1) for x in range(numOfAssets))
    # Tangency Portfolio

    if shortValid == True:
        opt_sharpe = sco.minimize(min_sharpe_ratio, initial_wts, method='SLSQP', constraints=cons)
    else:
        opt_sharpe = sco.minimize(min_sharpe_ratio, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)

    w = opt_sharpe.x
    assetWeight = pd.DataFrame(w, index=stocks, columns=['Weight'])
    return assetWeight


def global_minvol_weights(mu: "array-like", cov: "array-like", rf: "float", shortValid=True):
    """
    params:
    mu: asset average return, size:n x 1
    cov: portfolio covariance, size nxn
    rf: risk-free rate
    shortValid: whether short sale is allowed
    """

    def portfolio_stats(weights):
        weights = np.array(weights)[:, np.newaxis]
        port_rets = weights.T @ mu
        port_vols = np.sqrt(weights.T @ cov @ weights)

        return np.array([port_rets, port_vols, (port_rets - rf) / port_vols]).flatten()

    # Minimize the volatility
    def min_volatility(weights):
        return portfolio_stats(weights)[1]

    numOfAssets = len(mu)
    cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
    stocks = mu.index
    mu = np.array(mu)
    cov = np.array(cov)
    initial_wts = np.array(numOfAssets * [1. / numOfAssets])
    bnds = tuple((0, 1) for x in range(numOfAssets))
    # Tangency Portfolio

    if shortValid == True:
        opt_sharpe = sco.minimize(min_volatility, initial_wts, method='SLSQP', constraints=cons)
    else:
        opt_sharpe = sco.minimize(min_volatility, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)

    w = opt_sharpe.x
    assetWeight = pd.DataFrame(w, index=stocks, columns=['Weight'])
    return assetWeight