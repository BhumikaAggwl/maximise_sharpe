# 📊 Portfolio Optimization – Maximizing Sharpe Ratio  
### Problem 2.1 (Advanced) | Infinitrix Financial Mathematics 2025

---

## 🧠 Overview
This project implements a **mean-variance portfolio optimization model** that maximizes the **Sharpe Ratio** while enforcing budget and non-negativity constraints.  
It demonstrates the complete mathematical–to–computational pipeline:  
data acquisition → return/covariance estimation → optimization → sensitivity analysis → visualization.

---

## 🧮 Optimization Model
\[
\max_w \frac{w^T\mu - r_f}{\sqrt{w^T \Sigma w}}
\quad
\text{s.t. } \sum_i w_i = 1,\; w_i \ge 0
\]

where  

* \( w \) – portfolio weights  
* \( \mu \) – expected asset returns  
* \( \Sigma \) – covariance matrix of returns  
* \( r_f \) – risk-free rate  

Optimization solved with **Sequential Least Squares Programming (SLSQP)** from `scipy.optimize`.

---

## 🏗️ Repository Structure
```bash 
maximise_sharpe_ratio/
│
├── outputs/ # auto-saved plots & CSV summary
│ ├── efficient_frontier.png
│ ├── optimal_portfolio_pie.png
│ ├── sensitivity_rf.png
│ ├── sensitivity_shrinkage.png
│ └── portfolio_summary.csv
│
├── scripts/
│ └── run_optimization.py
│
├── src/
│ └── portopt/
│ ├── init.py
│ ├── data_loader.py # Yahoo + Alpha Vantage hybrid loader
│ ├── optimizer.py # Sharpe-ratio maximization
│ ├── sensitivity.py # rf & shrinkage analysis
│ ├── visualization.py # all non-blocking plots
│ └── …
│
├── tests/
│ └── test_optimizer.py
│
├── main.py # main pipeline (data→opt→plots)
├── requirements.txt
└── README.md
```
## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/maximise_sharpe_ratio.git
cd maximise_sharpe_ratio
python -m venv venv
source venv/bin/activate      # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```
## Requirements
```bash 
numpy
pandas
matplotlib
scipy
yfinance
alpha_vantage
```

## 🚀 Usage
```bash 
python main.py
```

Default assets: AAPL | MSFT | GOOGL | AMZN
Default period: 2023-01-01 → 2025-01-01

### 🗝️ Alpha Vantage API Key (fallback)
Yahoo Finance occasionally rate-limits requests.
To enable automatic fallback:

1) Get a free key → https://www.alphavantage.co/support/#api-key
2) Insert it in main.py
```bash 
API_KEY = "YOUR_ALPHA_VANTAGE_KEY"
```
3) The script will try Yahoo Finance once, then Alpha Vantage, then cached CSV.

## 📈 Outputs (Stored in /outputs)

| File                          | Description                                      |
| ----------------------------- | ------------------------------------------------ |
| **efficient_frontier.png**    | Efficient Frontier with optimal Sharpe portfolio |
| **optimal_portfolio_pie.png** | Asset allocation pie chart                       |
| **sensitivity_rf.png**        | Sharpe Ratio vs Risk-Free Rate                   |
| **sensitivity_shrinkage.png** | Sharpe Ratio vs Covariance Shrinkage Intensity   |
| **portfolio_summary.csv**     | Optimal weights + Sharpe ratio summary           |

## 🧪 Sensitivity Analyses
Risk-Free Rate Sensitivity
Shows how Sharpe Ratio declines as 
𝑟
𝑓
r
f
	​

 rises.

## Shrinkage Sensitivity
Tests portfolio robustness when the covariance matrix is shrunk toward its diagonal:

Σ shrink=(1−α)Σ+αdiag(Σ)

## 📊 Example Output
```bash 
========= OPTIMAL PORTFOLIO SUMMARY =========
AAPL: 42.3%
MSFT: 28.7%
GOOGL: 17.5%
AMZN: 11.5%
Expected Annual Return: 14.2%
Expected Volatility: 10.8%
Sharpe Ratio: 1.31
```

## 🧩 Mathematical Notes
1) Annualization assumes 252 trading days/year.
2) All weights constrained to [0, 1].
3) Covariance shrinkage α ∈ [0, 1] improves numerical stability for noisy data.

# maximise_sharpe
