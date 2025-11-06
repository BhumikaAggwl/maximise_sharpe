import pandas as pd
import numpy as np
from src.portopt.data_loader import get_data
from src.portopt.optimizer import optimize_portfolio, portfolio_performance
from src.portopt.visualization import plot_efficient_frontier, plot_portfolio_weights, plot_sensitivity
from src.portopt.sensitivity import sensitivity_rf
from scipy.optimize import minimize

def build_efficient_frontier(mean_returns, cov_matrix, points=100):
    results = {'returns': [], 'volatility': []}
    n = len(mean_returns)
    for r_target in np.linspace(min(mean_returns), max(mean_returns), points):
        cons = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) - r_target}
        )
        bounds = tuple((0, 1) for _ in range(n))
        res = minimize(lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))),
                       n * [1/n], method='SLSQP', bounds=bounds, constraints=cons)
        if res.success:
            results['returns'].append(r_target)
            results['volatility'].append(res.fun)
    return pd.DataFrame(results)

if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    data, returns = get_data(tickers)
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    rf = 0.02
    result = optimize_portfolio(mean_returns, cov_matrix, rf)
    ret, vol, sr = portfolio_performance(result.x, mean_returns, cov_matrix, rf)

    print("\nOptimal Weights:")
    for t, w in zip(tickers, result.x):
        print(f"{t}: {w:.2%}")
    print(f"\nPortfolio Return: {ret:.2%}")
    print(f"Portfolio Volatility: {vol:.2%}")
    print(f"Sharpe Ratio: {sr:.2f}")

    # Plots
    frontier = build_efficient_frontier(mean_returns, cov_matrix)
    plot_efficient_frontier(frontier, (ret, vol))
    plot_portfolio_weights(result.x, tickers)

    # Sensitivity Analysis
    rf_values = np.linspace(0, 0.05, 6)
    sharpe_values = sensitivity_rf(mean_returns, cov_matrix, rf_values)
    plot_sensitivity(rf_values, sharpe_values)
