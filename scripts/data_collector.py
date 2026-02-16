"""
Fetch OHLCV data for the configured asset.
- Crypto: Kraken via CCXT (symbol e.g. BTC/USDT).
- Stock:  NYSE/NASDAQ etc. via yfinance (symbol e.g. SPY, AAPL).
"""
import pandas as pd
from datetime import datetime, timedelta

from config import (
    ASSET,
    ASSET_TYPE,
    SYMBOL,
    TIMEFRAME,
    HISTORY_DAYS,
    HISTORICAL_DATA_PATH,
)


def fetch_crypto_ohlcv(symbol, timeframe="1h", since=None):
    """Fetch OHLCV from Kraken via CCXT."""
    import ccxt
    import time

    exchange = ccxt.kraken()
    all_data = []
    limit = 720  # Kraken's max per request

    while True:
        print(f"Fetching data since {since}...")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if len(ohlcv) == 0:
            break
        all_data.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        time.sleep(exchange.rateLimit / 1000)
        print(f"Fetched {len(ohlcv)} candles, total: {len(all_data)}")
        if len(ohlcv) < limit:
            break

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates(subset=["timestamp"])
    return df


def fetch_stock_ohlcv(symbol, interval="1h", period_days=365):
    """Fetch OHLCV from Yahoo Finance (NYSE, NASDAQ, etc.)."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("For stocks, install yfinance: pip install yfinance")

    # yfinance: 1h supports up to ~730 days; 1d supports long history
    end = datetime.now()
    start = end - timedelta(days=period_days)
    period_str = f"{period_days}d"

    print(f"Fetching {symbol} from {start.date()} to {end.date()} ({interval} bars)...")
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=start, end=end, interval=interval, auto_adjust=True)

    if hist.empty or len(hist) < 2:
        raise ValueError(f"No data returned for {symbol}. Check symbol and date range.")

    # Standardize columns: yfinance uses Date (daily) or Datetime (intraday) for index
    df = hist.reset_index()
    rename = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    # Index column can be "Date", "Datetime", or similar depending on yfinance version
    for date_name in ("Datetime", "Date", "index"):
        if date_name in df.columns:
            rename[date_name] = "timestamp"
            break
    else:
        # Fallback: first column is the former index (datetime)
        rename[df.columns[0]] = "timestamp"
    df = df.rename(columns=rename)
    # Keep only OHLCV + timestamp (drop Dividends, Stock Splits if present)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def fetch_all_ohlcv_data():
    """Fetch OHLCV based on config (crypto or stock)."""
    if ASSET_TYPE == "crypto":
        since_ms = int((datetime.now().timestamp() - HISTORY_DAYS * 24 * 60 * 60) * 1000)
        return fetch_crypto_ohlcv(SYMBOL, timeframe=TIMEFRAME, since=since_ms)
    elif ASSET_TYPE == "stock":
        return fetch_stock_ohlcv(SYMBOL, interval=TIMEFRAME, period_days=HISTORY_DAYS)
    else:
        raise ValueError(f'config.ASSET_TYPE must be "crypto" or "stock", got: {ASSET_TYPE!r}')


if __name__ == "__main__":
    print(f"Fetching {HISTORY_DAYS} days of {ASSET} ({ASSET_TYPE}) data ({TIMEFRAME})...")
    data = fetch_all_ohlcv_data()
    data.to_csv(HISTORICAL_DATA_PATH, index=False)
    print(f"\nSaved {len(data)} records to {HISTORICAL_DATA_PATH}")
    print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    print(data.head())
