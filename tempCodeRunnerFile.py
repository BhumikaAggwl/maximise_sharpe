# ============================================================
# main.py
# Portfolio Optimization - Maximizing Sharpe Ratio
# Problem 2.1 (Advanced) | Infinitrix Financial Mathematics
# ============================================================

import numpy as np
import pandas as pd

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.portopt import (
    get_data,
    optimize_portfolio,
    portfolio_performance,
    plot_efficient_frontier,
    plot_portfolio_weights,
    plot_sensitivity,
    sensitivity_rf
)

from src.portopt.sensitivity import sensitivity_rf
from scipy.optimize import minimize
import os

# ------------------------------------------------------------
# Helper: Build Efficient Frontier
# ------------------------------------------------------------
def build_efficient_frontier(mean_returns, cov_matrix, points=100):
    results = {'returns': [], 'volatility': []}
    n = len(mean_returns)
    for target_ret in np.linspace(min(mean_returns), max(mean_returns), points):
        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) - target_ret}
        )
        bounds = tuple((0, 1) for _ in range(n))
        res = minimize(lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))),
                       n * [1/n], method='SLSQP', bounds=bounds, constraints=constraints)
        if res.success:
            results['returns'].append(target_ret)
            results['volatility'].append(res.fun)
    return pd.DataFrame(results)


# ------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------
def main():
    os.makedirs("outputs", exist_ok=True)

    # Step 1: Choose assets and fetch data
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    data, returns = get_data(tickers)
    mean_returns = returns.mean() * 252       # annualized
    cov_matrix = returns.cov() * 252          # annualized
    rf = 0.02                                 # 2% risk-free rate

    # Step 2: Optimize portfolio for maximum Sharpe ratio
    result = optimize_portfolio(mean_returns, cov_matrix, rf)
    ret, vol, sr = portfolio_performance(result.x, mean_returns, cov_matrix, rf)

    print("\n========= OPTIMAL PORTFOLIO SUMMARY =========")
    for t, w in zip(tickers, result.x):
        print(f"{t}: {w:.2%}")
    print(f"Expected Annual Return: {ret:.2%}")
    print(f"Expected Volatility: {vol:.2%}")
    print(f"Sharpe Ratio: {sr:.3f}")

    # Step 3: Build efficient frontier
    frontier_df = build_efficient_frontier(mean_returns, cov_matrix)
    plot_efficient_frontier(frontier_df, (ret, vol))

    # Step 4: Portfolio weights pie chart
    plot_portfolio_weights(result.x, tickers)

    # Step 5: Sensitivity analysis (Sharpe vs rf)
    rf_values = np.linspace(0, 0.05, 6)
    sharpe_values = sensitivity_rf(mean_returns, cov_matrix, rf_values)
    plot_sensitivity(rf_values, sharpe_values)

    # Step 6: Save summary table
    summary = pd.DataFrame({
        'Ticker': tickers,
        'Weight': result.x
    })
    summary.loc['Portfolio', ['Ticker', 'Weight']] = ['Sharpe Ratio', sr]
    summary.to_csv('outputs/portfolio_summary.csv', index=False)
    print("\nResults saved to 'outputs/' folder.")


if __name__ == "__main__":
    main()
