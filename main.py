# ============================================================
# main.py
# Portfolio Optimization – Maximizing Sharpe Ratio
# Problem 2.1 (Advanced) | Infinitrix Financial Mathematics
# ============================================================

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
from src.portopt.data_loader import get_data
from src.portopt.optimizer import optimize_portfolio, portfolio_performance
from src.portopt.sensitivity import sensitivity_shrinkage
from src.portopt.visualization import plot_shrinkage_sensitivity
from src.portopt.visualization import (
    plot_efficient_frontier,
    plot_portfolio_weights,
    plot_sensitivity,
)
from src.portopt.sensitivity import sensitivity_rf
from scipy.optimize import minimize


# ------------------------------------------------------------
# Helper: Efficient Frontier Builder
# ------------------------------------------------------------
def build_efficient_frontier(mean_returns, cov_matrix, points=100):
    results = {"returns": [], "volatility": []}
    n = len(mean_returns)

    for target_ret in np.linspace(min(mean_returns), max(mean_returns), points):
        cons = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w: np.dot(w, mean_returns) - target_ret},
        )
        bounds = tuple((0, 1) for _ in range(n))
        res = minimize(
            lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))),
            n * [1 / n],
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
        )
        if res.success:
            results["returns"].append(target_ret)
            results["volatility"].append(res.fun)

    return pd.DataFrame(results)


# ------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------
def main():
    os.makedirs("outputs", exist_ok=True)

    # -------------------------------
    # 1️⃣ Configuration
    # -------------------------------
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    rf = 0.02  # risk-free rate
    API_KEY = "9SXYEXUYI1CTA6DA"  # ← put your key here for fallback

    print("🔹 Fetching data for:", tickers)

    # -------------------------------
    # 2️⃣ Fetch Data (with fallback)
    # -------------------------------
    try:
        data, returns = get_data(
            tickers,
            start="2023-01-01",
            end="2025-01-01",
            api_key=API_KEY,
        )
    except Exception as e:
        print(f"❌ Data fetch failed: {e}")
        return

    if data.empty or returns.empty:
        print("⚠️ No valid data retrieved. Check your API key or wait for Yahoo unblock.")
        return

    print(f"✅ Data loaded: {data.shape[0]} days × {data.shape[1]} assets")

    # -------------------------------
    # 3️⃣ Compute Mean & Covariance
    # -------------------------------
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    # -------------------------------
    # 4️⃣ Optimize Portfolio
    # -------------------------------
    try:
        result = optimize_portfolio(mean_returns, cov_matrix, rf)
        ret, vol, sr = portfolio_performance(result.x, mean_returns, cov_matrix, rf)
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return

    # -------------------------------
    # 5️⃣ Print Results
    # -------------------------------
    print("\n========= OPTIMAL PORTFOLIO SUMMARY =========")
    for t, w in zip(tickers, result.x):
        print(f"{t}: {w:.2%}")
    print(f"Expected Annual Return: {ret:.2%}")
    print(f"Expected Volatility: {vol:.2%}")
    print(f"Sharpe Ratio: {sr:.2f}")

    # -------------------------------
    # 6️⃣ Generate Outputs
    # -------------------------------
    # -------------------------------
# 6️⃣ Generate Outputs
# -------------------------------
    try:
        frontier_df = build_efficient_frontier(mean_returns, cov_matrix)
        plot_efficient_frontier(frontier_df, (ret, vol))
        plot_portfolio_weights(result.x, tickers)

    # --- Sensitivity to Risk-Free Rate ---
        rf_values = np.linspace(0, 0.05, 6)
        sharpe_values = sensitivity_rf(mean_returns, cov_matrix, rf_values)
        plot_sensitivity(rf_values, sharpe_values)

  

        shrink_df = sensitivity_shrinkage(mean_returns, cov_matrix, rf=rf)
        plot_shrinkage_sensitivity(shrink_df)

    except Exception as e:
        print(f"⚠️ Plotting skipped due to error: {e}")


    # -------------------------------
    # 7️⃣ Save Summary
    # -------------------------------
    summary = pd.DataFrame({"Ticker": tickers, "Weight": result.x})
    summary.loc["Portfolio"] = ["Sharpe Ratio", sr]
    summary.to_csv("outputs/portfolio_summary.csv", index=False)
    print("\n📁 Results saved to 'outputs/' folder.")


# ------------------------------------------------------------
if __name__ == "__main__":
    main()
