import numpy as np
from scipy.optimize import minimize

def portfolio_performance(weights, mean_returns, cov_matrix, rf=0.0):
    weights = np.array(weights)
    port_return = np.dot(weights, mean_returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (port_return - rf) / port_volatility
    return port_return, port_volatility, sharpe_ratio


def neg_sharpe_ratio(weights, mean_returns, cov_matrix, rf=0.0):
    return -portfolio_performance(weights, mean_returns, cov_matrix, rf)[2]


def optimize_portfolio(mean_returns, cov_matrix, rf=0.0):
    """
    Solve max Sharpe ratio optimization:
    max (wTμ - rf) / sqrt(wTΣw)
    s.t. Σ wi = 1, wi ≥ 0
    """
    n = len(mean_returns)
    args = (mean_returns, cov_matrix, rf)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    init_guess = n * [1. / n]
    result = minimize(neg_sharpe_ratio, init_guess, args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result
