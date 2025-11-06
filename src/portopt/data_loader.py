"""
data_loader.py
---------------
Unified data loader for financial time-series.

1️⃣  Try Yahoo Finance once
2️⃣  If that fails or returns empty → fall back to Alpha Vantage
3️⃣  If both fail → use local cached CSV (if available)
"""

import os
import pandas as pd
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import time


from alpha_vantage.timeseries import TimeSeries
import pandas as pd, time

def _get_data_alphavantage(tickers, api_key, delay=12):
    """Free-tier Alpha Vantage fallback using unadjusted daily closes."""
    ts = TimeSeries(key=api_key, output_format="pandas")
    all_prices = pd.DataFrame()

    for t in tickers:
        print(f"🔄 Fetching {t} (free endpoint)…")
        data, _ = ts.get_daily(symbol=t, outputsize="full")   # 👈 free endpoint
        data = data.rename(columns={"4. close": t})
        all_prices[t] = data[t].iloc[::-1]   # ascending order
        time.sleep(delay)                    # respect 5 calls/min limit

    all_prices = all_prices.dropna()
    returns = all_prices.pct_change().dropna()
    return all_prices, returns


def get_data(
    tickers,
    start="2023-01-01",
    end="2025-01-01",
    api_key=None,
    cache_dir="data_cache",
):
    """
    Fetch adjusted closing prices and returns.

    • Try Yahoo Finance once
    • Fall back to Alpha Vantage if Yahoo fails
    • Use cached file if both unavailable
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "cached_prices.csv")

    # -------------------------------------------------------
    # Try Yahoo Finance (only once)
    # -------------------------------------------------------
    try:
        print("📥 Downloading via Yahoo Finance…")
        data = yf.download(tickers, start=start, end=end, progress=False, threads=False)
        if data.empty:
            raise ValueError("Yahoo Finance returned empty data")

        # Handle multi-ticker or single-ticker format
        if isinstance(data.columns, pd.MultiIndex):
            if "Adj Close" in data.columns.levels[0]:
                data = data["Adj Close"]
            elif "Close" in data.columns.levels[0]:
                data = data["Close"]
        else:
            if "Adj Close" in data.columns:
                data = data[["Adj Close"]]
            elif "Close" in data.columns:
                data = data[["Close"]]
            else:
                raise KeyError("No valid price column found.")

        data = data.dropna()
        returns = data.pct_change().dropna()
        data.to_csv(cache_file)
        print("✅ Yahoo Finance download successful.")
        return data, returns

    except Exception as e:
        print(f"⚠️ Yahoo Finance failed: {e}")

    # -------------------------------------------------------
    # Fallback: Alpha Vantage (if API key provided)
    # -------------------------------------------------------
    if api_key:
        try:
            print("🔁 Falling back to Alpha Vantage…")
            data, returns = _get_data_alphavantage(tickers, api_key)
            data.to_csv(cache_file)
            return data, returns
        except Exception as e:
            print(f"❌ Alpha Vantage fallback failed: {e}")

    # -------------------------------------------------------
    # Last resort: use cached data if available
    # -------------------------------------------------------
    if os.path.exists(cache_file):
        print("⚙️ Using cached data instead.")
        data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        returns = data.pct_change().dropna()
        return data, returns

    raise ConnectionError("❌ All data sources failed and no cache available.")
