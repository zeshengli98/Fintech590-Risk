import numpy as np
import statsmodels.api as sm
import pandas as pd
import scipy.optimize as sco
from risk_management import ES
def update_weights(weights, returns):
    """
    Update the weights of each asset in a portfolio given the initial
    weight and returns. The initial weight and returns starts at the
    same period.
    params:
        - weights: DataFrame, shape(n,)
        - returns: DataFrame, shape(t, n)
    return:
        - updated_weights: DataFrame, shape(t, n)
    """
    latest_weights = weights.values[0].copy()
    updated_weights = np.zeros(shape=(returns.shape[0], len(latest_weights)))
    for i in range(returns.shape[0]):
        updated_weights[i, :] = latest_weights
        latest_weights *= (1 + returns.iloc[i, :])
        latest_weights /= sum(latest_weights)

    return pd.DataFrame(updated_weights ,index = returns.index ,columns = returns.columns)

def cal_carino_k(weighted_rets_t):
    R = (weighted_rets_t +1).prod(axis=0 ) -1
    K = np.log( 1 +R ) /R
    kt = np.log( 1 +weighted_rets_t ) /( K *weighted_rets_t)
    return kt


def return_attribution(weighted_rets):
    weighted_rets_t = weighted_rets.sum(axis=1)
    return cal_carino_k(weighted_rets_t) @ weighted_rets

def risk_attribution(weighted_rets):

    weighted_rets_t = weighted_rets.sum(axis=1)
    port_sigma = weighted_rets_t.std()
    n = weighted_rets.shape[1]

    risk_attribution = pd.Series()
    for st in weighted_rets.columns:
        # ri = alpha + sum(beta * rt) + error
        model = sm.OLS(weighted_rets[st], sm.add_constant(weighted_rets_t))
        results = model.fit()
        risk_attribution[st] = results.params.values[1] * port_sigma
    return risk_attribution


def cal_port_vol(weights, cov):
    w = np.array(weights).flatten()
    port_std = np.sqrt(w @ cov @ w)
    return port_std


def cal_component_std(weights, cov):
    w = np.array(weights).flatten()
    port_std = cal_port_vol(weights, cov)
    csd = w * (cov @ w) / port_std
    return csd


def risk_budget(weights, cov):
    w = np.array(weights).flatten()
    port_std = cal_port_vol(weights, cov)
    csd = cal_component_std(weights, cov)
    return csd / port_std


def cal_csd_sse(weights, cov):
    w = np.array(weights).flatten()
    port_std = cal_port_vol(weights, cov)
    csd = cal_component_std(weights, cov)
    csd_ = csd - np.mean(csd)
    return sum(csd_ ** 2)


def risk_parity_weights(cov):
    numOfAssets = cov.shape[0]
    initial_wts = np.array(numOfAssets * [1. / numOfAssets])
    cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
    cov = np.array(cov)
    bnds = tuple((0, 1) for x in range(numOfAssets))
    opt_wts = sco.minimize(lambda w: 1e5 * cal_csd_sse(w, cov), initial_wts, bounds=bnds, constraints=cons)
    return opt_wts.x


def cal_port_es(weights, rets):
    w = np.array(weights).flatten()
    return ES.ES(rets @ w)


def cal_component_es(weights, rets):
    w = np.array(weights).flatten()
    e = 1e-6
    numOfAssets = len(weights)
    es = cal_port_es(w, rets)
    ces = np.zeros(numOfAssets)

    for i in range(numOfAssets):
        weight_i = w[i]
        w[i] += e
        ces[i] = weight_i * (cal_port_es(w, rets) - es) / e
        w[i] -= e

    return ces


def cal_ces_sse(weights, rets):
    ces = cal_component_es(weights, rets)
    ces_ = ces - np.mean(ces)
    return ces_ @ ces_.T


def risk_parity_es_weights(rets):
    numOfAssets = rets.shape[1]
    initial_wts = np.array(numOfAssets * [1. / numOfAssets])
    cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
    rets = np.array(rets)
    bnds = tuple((0, 1) for x in range(numOfAssets))
    opt_wts = sco.minimize(lambda w: 1e5 * cal_ces_sse(w, rets), initial_wts, bounds=bnds, constraints=cons)
    return opt_wts.x