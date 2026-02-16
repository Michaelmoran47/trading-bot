"""
Trading bot configuration — set asset and type in this single file.
Supports: any NYSE (and other US) stock, or any crypto available on Kraken.
"""

# -----------------------------------------------------------------------------
# ASSET CONFIGURATION — edit only this section
# -----------------------------------------------------------------------------

# Asset type: "crypto" or "stock"
#   - crypto: fetches via CCXT (Kraken). Use SYMBOL format expected by exchange (e.g. BTC/USDT).
#   - stock:  fetches via yfinance (NYSE, NASDAQ, etc.). Use ticker symbol (e.g. SPY, AAPL).
ASSET_TYPE = "stock"

# Display name: used in labels, filenames, and messages (e.g. "BTC", "SPY", "AAPL").
ASSET = "F"

# Symbol for the data source:
#   - For crypto: exchange pair, e.g. "BTC/USDT", "ETH/USDT".
#   - For stock:  same as ticker, e.g. "SPY", "AAPL".
SYMBOL = "F"

# Candle timeframe: "1h" (hourly) or "1d" (daily).
#   - Crypto: 1h typical; Kraken supports both.
#   - Stock:  1h = up to ~2 years of history; 1d = many years.
TIMEFRAME = "1h"

# How far back to fetch (days). Ignored for crypto (uses max available per request).
HISTORY_DAYS = 365

# -----------------------------------------------------------------------------
# Derived paths (used by scripts; no need to edit)
# -----------------------------------------------------------------------------
_asset_lower = ASSET.lower()
HISTORICAL_DATA_PATH = f"data/{_asset_lower}_historical_data.csv"
FEATURES_PATH = f"data/{_asset_lower}_features.csv"
MODEL_PATH = f"models/trained_model_{_asset_lower}.pkl"
