import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # use non-GUI backend
import matplotlib.pyplot as plt

def plot_efficient_frontier(frontier_df, optimal_point):
    plt.figure(figsize=(8,6))
    plt.plot(frontier_df['volatility'], frontier_df['returns'], label='Efficient Frontier')
    plt.scatter(optimal_point[1], optimal_point[0], color='r', marker='*', s=200, label='Optimal Portfolio')
    plt.xlabel('Volatility (σp)')
    plt.ylabel('Expected Return (Rp)')
    plt.title('Efficient Frontier')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/efficient_frontier.png')
    plt.close()  # ✅ prevent blocking


def plot_portfolio_weights(weights, tickers):
    plt.figure(figsize=(6,6))
    plt.pie(weights, labels=tickers, autopct='%1.1f%%', startangle=140)
    plt.title('Optimal Portfolio Allocation')
    plt.tight_layout()
    plt.savefig('outputs/optimal_portfolio_pie.png')
    plt.close()  # ✅ prevent blocking


def plot_sensitivity(rf_values, sharpe_values):
    plt.figure(figsize=(7,5))
    plt.plot(rf_values, sharpe_values, 'o-', lw=2)
    plt.xlabel('Risk-Free Rate')
    plt.ylabel('Sharpe Ratio')
    plt.title('Sensitivity of Sharpe Ratio to Risk-Free Rate')
    plt.tight_layout()
    plt.savefig('outputs/sensitivity_rf.png')
    plt.close()  # ✅ prevent blocking


def plot_shrinkage_sensitivity(df):
    plt.figure(figsize=(7,5))
    plt.plot(df["Shrinkage"], df["Sharpe"], 'o-', lw=2)
    plt.xlabel("Shrinkage Intensity (α)")
    plt.ylabel("Sharpe Ratio")
    plt.title("Sensitivity of Sharpe Ratio to Covariance Shrinkage")
    plt.tight_layout()
    plt.savefig("outputs/sensitivity_shrinkage.png")
    plt.close()  # ✅ prevent blocking
