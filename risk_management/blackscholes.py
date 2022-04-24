
import numpy as np
import pandas as pd
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


class BSOption:
    def __init__(self, S, K, T, r, q, sigma, call):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.sigma = sigma
        self.call = call
        self.d1 = (np.log(S / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        self.d2 = self.d1 - sigma * np.sqrt(T)
        self.Delta = self.delta()
        self.Gamma = self.gamma()
        self.Vega = self.vega()
        self.Theta = self.theta()
        self.Rho = self.rho()
        self.Phi = self.phi()
        self.Greek = self.greek()
        self.Price = self.black_scholes()

    def black_scholes(self):
        if self.call == True:
            return self.S * np.exp(-self.q * self.T) * norm.cdf(self.d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(
                self.d2)
        else:
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - self.S * np.exp(
                -self.q * self.T) * norm.cdf(-self.d1)

    #  greeks for GBSM
    def delta(self):
        factor = np.exp((-self.q * self.T))
        if self.call == True:
            return factor * norm.cdf(self.d1)
        else:
            return factor * (norm.cdf(self.d1) - 1)

    def gamma(self):
        return np.exp(-self.q * self.T) * norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        return self.S * np.exp(-self.q * self.T) * norm.pdf(self.d1) * np.sqrt(self.T) / 100

    def theta(self):
        if self.call == True:
            opt = 1
        else:
            opt = -1
        df = np.exp(-self.r * self.T)

        dfq = np.exp(-self.q * self.T)
        tmptheta = -0.5 * self.S * dfq * norm.pdf(self.d1) * self.sigma / np.sqrt(
            self.T) + opt * self.q * self.S * dfq * norm.cdf(self.d1) - opt * self.r * self.K * df * norm.cdf(
            opt * self.d2)
        return tmptheta / 365

    def rho(self):
        sign = 1 if self.call == True else -1
        df = np.exp(-self.r * self.T)
        return sign * self.T * self.K * df * norm.cdf(sign * self.d2) / 100

    def phi(self):
        sign = 1 if self.call == True else -1
        dfq = np.exp(-self.q * self.T)
        return -sign * self.T * self.S * dfq * norm.cdf(sign * self.d1) / 100

    def greek(self):
        opt = 'Call' if self.call == True else 'Put'
        greek_frame = pd.DataFrame(np.array([self.Delta, self.Gamma, self.Vega, self.Theta, self.Rho, self.Phi]),
                                   index=['delta', 'gamma', 'vega', 'theta', 'rho', 'phi'], columns=['GBSM ' + opt])
        return greek_frame


class FDOption:
    def __init__(self, S, K, T, r, q, sigma, call):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.sigma = sigma
        self.call = call
        self.d1 = (np.log(S / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        self.d2 = self.d1 - sigma * np.sqrt(T)
        self.Delta = self.delta()
        self.Gamma = self.gamma()
        self.Vega = self.vega()
        self.Theta = self.theta()
        self.Rho = self.rho()
        self.Phi = self.phi()
        self.Greek = self.greek()
        self.Price = self.black_scholes()

    def black_scholes(self):
        if self.call == True:
            return self.S * np.exp(-self.q * self.T) * norm.cdf(self.d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(
                self.d2)
        else:
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - self.S * np.exp(
                -self.q * self.T) * norm.cdf(-self.d1)

    # Implement a finite difference derivative calculation.
    def delta(self, ds=0.01):
        u = black_scholes(self.S + ds, self.K, self.T, self.r, self.q, self.sigma, self.call)
        d = black_scholes(self.S - ds, self.K, self.T, self.r, self.q, self.sigma, self.call)
        return 0.5 * (u - d) / ds

    def gamma(self, ds=0.01):

        u = black_scholes(self.S + ds, self.K, self.T, self.r, self.q, self.sigma, self.call)
        m = black_scholes(self.S, self.K, self.T, self.r, self.q, self.sigma, self.call)
        d = black_scholes(self.S - ds, self.K, self.T, self.r, self.q, self.sigma, self.call)
        return (u - 2 * m + d) / ds / ds

    def vega(self, dsig=0.001):
        u = black_scholes(self.S, self.K, self.T, self.r, self.q, self.sigma + dsig, self.call)
        d = black_scholes(self.S, self.K, self.T, self.r, self.q, self.sigma - dsig, self.call)
        return 0.5 * (u - d) / dsig / 100

    def theta(self, dt=0.001):
        u = black_scholes(self.S, self.K, self.T - dt, self.r, self.q, self.sigma, self.call)
        d = black_scholes(self.S, self.K, self.T + dt, self.r, self.q, self.sigma, self.call)
        return 0.5 * (u - d) / dt / 365

    def rho(self, dr=0.001):
        u = black_scholes(self.S, self.K, self.T, self.r + dr, self.q, self.sigma, self.call)
        d = black_scholes(self.S, self.K, self.T, self.r - dr, self.q, self.sigma, self.call)
        return 0.5 * (u - d) / dr / 100

    def phi(self, db=0.001):
        u = black_scholes(self.S, self.K, self.T, self.r, self.q + db, self.sigma, self.call)
        d = black_scholes(self.S, self.K, self.T, self.r, self.q - db, self.sigma, self.call)
        return 0.5 * (u - d) / db / 100

    def greek(self):
        opt = 'Call' if self.call == True else 'Put'
        greek_frame = pd.DataFrame(np.array([self.Delta, self.Gamma, self.Vega, self.Theta, self.Rho, self.Phi]),
                                   index=['delta', 'gamma', 'vega', 'theta', 'rho', 'phi'],
                                   columns=['Finite Diff ' + opt])
        return greek_frame


## American option pricing
def BinomialTree(Otype, S0, K, r, sigma, T, N=2000, american='false', divi=[[], []]):
    # we improve the previous tree by generalization of payoff
    def expiCall(stock, strike):
        return (np.maximum(stock - strike, 0))

    def expiPut(stock, strike):
        return (np.maximum(strike - stock, 0))

    def earlyCall(Option, stock, strike):
        return (np.maximum(stock - strike, Option))

    def earlyEuro(Option, stock, strike):
        return (np.maximum(Option, 0))

    def earlyPut(Option, stock, strike):
        return (np.maximum(strike - stock, Option))

    if Otype == 'C':
        expi = expiCall
        early = earlyCall
    else:
        expi = expiPut
        early = earlyPut

    if american == 'false':
        early = earlyEuro

    # so no matter what we calculate expiration value with the expi function and
    # early exercise with the function early

    # calculate delta T
    deltaT = float(T) / N
    dividends = [[], []]

    # 2 columns vector, one for amount one for time.
    # we make sure we don't take into account dividend happening after expiration
    if (np.size(divi) > 0 and divi[0][0] < T):
        lastdiv = np.nonzero(np.array(divi[0][:]) <= T)[0][-1]
        dividends[0] = divi[0][:lastdiv + 1]
        dividends[1] = divi[1][:lastdiv + 1]

        # Transfrom the dividend date into a step
    if np.size(dividends) > 0:
        dividendsStep = np.floor(np.multiply(dividends[0], 1 / deltaT))
    else:
        dividendsStep = []

        # get present value of the dividend
    if np.size(dividends) > 0:
        pvdividends = np.sum(np.multiply(dividends[1], np.exp(np.multiply(dividendsStep, -r * deltaT))))
    else:
        pvdividends = 0

    # we apply the usuall hull method
    S0 = S0 - pvdividends
    currentDividend = 0

    # up and down factor will be constant for the tree so we calculate outside the loop
    u = np.exp(sigma * np.sqrt(deltaT))
    d = 1.0 / u

    # to work with vector we need to init the arrays using numpy
    fs = np.asarray([0.0 for i in range(N + 1)])

    # we need the stock tree for calculations of expiration values
    fs2 = np.asarray([(S0 * u ** j * d ** (N - j)) for j in range(N + 1)])

    # we vectorize the strikes as well so the expiration check will be faster
    fs3 = np.asarray([float(K) for i in range(N + 1)])

    # rates are fixed so the probability of up and down are fixed.
    # this is used to make sure the drift is the risk free rate
    a = np.exp(r * deltaT)
    p = (a - d) / (u - d)
    oneMinusP = 1.0 - p

    # Compute the leaves, f_{N, j}
    fs[:] = expi(fs2, fs3)

    # calculate backward the option prices
    for i in range(N - 1, -1, -1):
        fs[:-1] = np.exp(-r * deltaT) * (p * fs[1:] + oneMinusP * fs[:-1])
        fs2[:] = fs2[:] * u
        currentDividend = currentDividend / a

        # we add back the dividends in current dividends as we move backward, useful
        # only for american options
        if (i in dividendsStep):
            div = dividends[1]
            div = div[np.nonzero(dividendsStep == (i))[0][0]]
            currentDividend = currentDividend + div

        fs[:] = early(fs[:], fs2[:] + currentDividend, fs3[:])

    return fs[0]


class BinomialTreeOption:
    def __init__(self, Otype, S0, K, r, sigma, T, N=2000, american='false', divi=[[], []]):
        self.Otype = Otype
        self.S0 = S0
        self.T = T
        self.r = r
        self.K = K
        self.sigma = sigma
        self.N = N
        self.american = american
        self.divi = divi
        self.Delta = self.delta()
        self.Gamma = self.gamma()
        self.Vega = self.vega()
        self.Theta = self.theta()
        self.Rho = self.rho()
        self.Phi = self.phi()
        self.Greek = self.greek()
        self.Price = self.cal_price()

    def cal_price(self):
        return BinomialTree(self.Otype, self.S0, self.K, self.r, self.sigma, self.T, self.N, self.american, self.divi)

    # Implement a finite difference derivative calculation.
    def delta(self, ds=0.01):
        u = BinomialTree(self.Otype, self.S0 + ds, self.K, self.r, self.sigma, self.T, self.N, self.american, self.divi)
        d = BinomialTree(self.Otype, self.S0 - ds, self.K, self.r, self.sigma, self.T, self.N, self.american, self.divi)
        return 0.5 * (u - d) / ds

    def gamma(self, ds=1):
        u = BinomialTree(self.Otype, self.S0 + ds, self.K, self.r, self.sigma, self.T, self.N, self.american, self.divi)
        m = BinomialTree(self.Otype, self.S0, self.K, self.r, self.sigma, self.T, self.N, self.american, self.divi)
        d = BinomialTree(self.Otype, self.S0 - ds, self.K, self.r, self.sigma, self.T, self.N, self.american, self.divi)
        return (u - 2 * m + d) / ds / ds

    def vega(self, dsig=0.001):
        u = BinomialTree(self.Otype, self.S0, self.K, self.r, self.sigma + dsig, self.T, self.N, self.american,
                         self.divi)
        d = BinomialTree(self.Otype, self.S0, self.K, self.r, self.sigma - dsig, self.T, self.N, self.american,
                         self.divi)
        return 0.5 * (u - d) / dsig / 100

    def theta(self, dt=0.001):
        u = BinomialTree(self.Otype, self.S0, self.K, self.r, self.sigma, self.T - dt, self.N, self.american, self.divi)
        d = BinomialTree(self.Otype, self.S0, self.K, self.r, self.sigma, self.T + dt, self.N, self.american, self.divi)
        return 0.5 * (u - d) / dt / 365

    def rho(self, dr=0.001):
        u = BinomialTree(self.Otype, self.S0, self.K, self.r + dr, self.sigma, self.T, self.N, self.american, self.divi)
        d = BinomialTree(self.Otype, self.S0, self.K, self.r - dr, self.sigma, self.T, self.N, self.american, self.divi)
        return 0.5 * (u - d) / dr / 100

    def phi(self, ddiv=0.001):
        div_u = [self.divi[0], [1 + ddiv]]
        div_d = [self.divi[0], [1 - ddiv]]
        u = BinomialTree(self.Otype, self.S0, self.K, self.r, self.sigma, self.T, self.N, self.american, div_u)
        d = BinomialTree(self.Otype, self.S0, self.K, self.r, self.sigma, self.T, self.N, self.american, div_d)
        return 0.5 * (u - d) / ddiv

    def greek(self):
        opt = 'Call' if self.Otype == 'C' else 'Put'
        greek_frame = pd.DataFrame(np.array([self.Delta, self.Gamma, self.Vega, self.Theta, self.Rho, self.Phi]),
                                   index=['delta', 'gamma', 'vega', 'theta', 'rho', 'dividen sensitivity'],
                                   columns=['Binomial Tree ' + opt])
        return greek_frame