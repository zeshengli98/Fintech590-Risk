# Fintech590-Risk

Risk_management package concludes a quantitative risk management library
For weekly project:
- [week02](week02/week02.ipynb):OLS, multivariate normal; ARMA, acf, pacf
- [week03](week03/week03.ipynb):PCA, cov, weighted cov; Cholesky Fatoriaztion, find nearest psd; PCA simulation
- [week04](week04/week04.ipynb):Monte Carlo simulation; VaR; portfolio VaR, delta normal
- [week05](week05/week05.ipynb):ES, T by MLE fit; portfolio t simulation
- [week06](week06/week06.ipynb):Black-scholes; implied vol; option portfolio; option portfolio VaR, ES
- [week07](week07/week07.ipynb):Greek; American option; Fama French 3 factors; efficient frontier
- [week08](week08/week08.ipynb):Ex-post, Ex-Ante attribution; return attribution; risk attribution; risk budgeting, risk parity

## library instruction
[Risk Library](risk_management)
### [Simulation](risk_management/simulation.py)
- PriceSimulator(S, mu, sigma) # generate a price simulator
  - PriceSimulator.simulate(method, path, known_step, nOfDarws)
- Return_norm_simulation(array, size) # simulation for norm distribution
- Return_T_params(array) # T Distribution by MLE fit
- copula_t_simulation(ret, nOfDarws) # give a ret for several stocks dataframe and return simulated copula draws
- simulate_path(s0, mu, sigma, horizon, timesteps, n_sims) # use sde discretization method doing simulation
### [VaR](risk_management/VaR.py)
- return_calculate(data, reset_index = True) # data: dataframe
- VaR_normal(ret, alpha = 0.05)
- VaR_EWnorm(ret, alpha = 0.05, lamb=0.95)
- VaR_T(ret, alpha = 0.05)
- VaR_Historic(ret, alpha = 0.05)
### [ES](risk_management/ES.py)
- ES(ret_arr, alpha = 0.05)
### [Attribution](risk_management/attribution.py)
- update_weights(weights, returns) # give returns dataframe and weights for each stock returning updated weights
- cal_carino_k(weighted_rets_t)
- return_attribution(weighted_rets) # after cal updated weighted returns, give the new returns and back return attribution
- risk_attribution(weighted_rets) # return risk attribution
- risk_parity_weights(cov) # give portfolio cov and return risk parity weights
- risk_parity_es_weights(rets) # give portfolio simulated rets or historical rets and return risk parity on ES weights
### [Efficient Frontier](risk_management/effFrontier.py)
- MinRiskPortfolio(mu, cov, m): give portfolio mu, cov and target rets m; back (port weights, target_vol)
- effFrontier(mu, cov, scope, rf, shortValid=True): efficient frontier simulator
- tangency_weights(mu, cov, rf, shortValid=True): find tangency portfolio and return its weights
- global_minvol_weights(mu, cov, rf, shortValid=True): find global minimum risk portfolio weights
### [Black Scholes](risk_management/blackscholes.py)
- black_scholes(S,K,T,r,q,sigma,call=True): cal euro option price
- implied_volatility(S,K,T,r,q,opt_price,call=True): cal implied vol
- BSOption( S, K, T, r, q, sigma, call): BS option class
- FDOption( S, K, T, r, q, sigma, call): BS option class
- BinomialTree(Otype, S0, K, r, sigma, T, N=2000, american='false', divi=[[], []]): American option pricing
- BinomialTreeOption(Otype, S0, K, r, sigma, T, N=2000, american='false', divi=[[], []])) American option binomial tree pricing
### [psd](risk_management/psd.py)
- chol_psd(A):Performs a Cholesky decomposition of a matrix, the matrix should be a symmetric and PD matrix. return: the lower triangle matrix.
- near_psd(A)
- Frobenius_norm(A)
- Higham_near_psd(A)
### [cov](risk_management/cov.py)
- weighted_cov(lamb, df): lamb is weight param, the dataframe format, demanding the time series is ascending
