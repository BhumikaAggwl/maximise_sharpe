import numpy as np
import pandas as pd
from .optimizer import optimize_portfolio, portfolio_performance

def sensitivity_rf(mean_returns, cov_matrix, rf_values):
    """
    Compute Sharpe Ratios for different risk-free rates.
    """
    sharpe_values = []
    for rf in rf_values:
        res = optimize_portfolio(mean_returns, cov_matrix, rf)
        _, _, sr = portfolio_performance(res.x, mean_returns, cov_matrix, rf)
        sharpe_values.append(sr)
    return sharpe_values


def sensitivity_shrinkage(mean_returns, cov_matrix, rf=0.0, shrink_values=None):
    """
    Analyze sensitivity of portfolio Sharpe Ratio to covariance shrinkage.
    Shrinkage = blending sample covariance with diagonal matrix.
    """
    if shrink_values is None:
        shrink_values = np.linspace(0, 1, 6)

    sharpe_values = []
    for alpha in shrink_values:
        # shrinkage covariance: (1-alpha)*S + alpha*diag(S)
        shrunk_cov = (1 - alpha) * cov_matrix + alpha * np.diag(np.diag(cov_matrix))
        res = optimize_portfolio(mean_returns, shrunk_cov, rf)
        _, _, sr = portfolio_performance(res.x, mean_returns, shrunk_cov, rf)
        sharpe_values.append(sr)
    return pd.DataFrame({"Shrinkage": shrink_values, "Sharpe": sharpe_values})
