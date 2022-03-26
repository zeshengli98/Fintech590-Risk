from scipy.stats import norm,t
import numpy as np
import pandas as pd
import scipy.optimize as optimize
from risk_management import cov
'''
data: 1-column DataFrame
'''
def return_calculate(data, reset_index = True):
    if reset_index == True:
        data = data.pct_change().dropna()
        data = data.reset_index()
        return data.drop('index', axis=1)
    else:
        return data.pct_change().dropna()

def VaR_normal(ret, alpha = 0.05):
    mean = ret.mean()
    sigma = ret.std()
    VaR_norm = -(norm.ppf(alpha)*sigma + mean)
    return VaR_norm[0]

def VaR_EWnorm(ret, alpha = 0.05, lamb=0.95):
    sigma = np.sqrt(cov.weighted_cov(lamb, ret)[0])
    mean = ret.mean()
    VaR_EWnorm = -(norm.ppf(alpha) * sigma + mean)
    return VaR_EWnorm[0]


def VaR_T(ret, alpha = 0.05):
    def t_log_lik(par_vec, x):
        lik = -np.log(t(df=par_vec[0],loc =par_vec[1], scale=par_vec[2]).pdf(x)).sum()
        return lik

    cons = ({'type': 'ineq', 'fun': lambda x: x[0] - 2},
            {'type': 'ineq', 'fun': lambda x: x[2]})
    df, mean, scale = optimize.minimize(fun=t_log_lik,
                                        x0=[2, np.array(ret).mean(), np.array(ret).std()],
                                        constraints=cons,
                                        args=(np.array(ret))).x
    VaR_t = -(scale * t.ppf(alpha, df) + mean)
    return VaR_t

def VaR_Historic(ret, alpha = 0.05):
    return -np.percentile(ret,q = alpha*100)


if __name__== "__main__":
    data = pd.read_csv("../problem1.csv")
    VaR1 = VaR_normal(data)
    VaR2 = VaR_EWnorm(data)
    VaR3 = VaR_T(data)
    VaR4 = VaR_Historic(data)

    print(VaR1,VaR2,VaR3,VaR4)
