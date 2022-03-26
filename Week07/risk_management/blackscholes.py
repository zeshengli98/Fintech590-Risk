
import numpy as np
from scipy.stats import norm,kstest,shapiro,t
from scipy.optimize import fsolve

def black_scholes(S,K,T,r,q,sigma,call=True):
    d1 = (np.log(S/K)+(r-q+sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1-sigma*np.sqrt(T)
    if call==True:
        return S*np.exp(-q*T)*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2)-S*np.exp(-q*T)*norm.cdf(-d1)

def implied_volatility(S,K,T,r,q,opt_price,call=True):
    volatility = lambda x: black_scholes(S,K,T,r,q,x,call) - opt_price
#     def eq(sigma):
#         return black_scholes(S,K,T,r,q,sigma,call=True) - opt_price
    return fsolve(volatility, x0 = 0.5)[0]