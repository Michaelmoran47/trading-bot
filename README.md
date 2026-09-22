# Trading Bot

An ML-driven trading research project: collect historical price data, engineer
technical-indicator features, train a Random Forest classifier to predict
next-period direction, backtest it, and (optionally) run it against Alpaca's
paper-trading API on a schedule.

This is a research/paper-trading prototype, not a production trading system.
See [Known limitations](#known-limitations) before trusting its output.

## How it's organized

A single file, `config.py`, drives the whole pipeline: pick an asset
(crypto via Kraken, or a US stock via Yahoo Finance) and every script below
reads/writes from paths derived from that choice.

```
config.py              # asset/timeframe settings — edit this to switch assets
scripts/
  data_collector.py    # fetch OHLCV history -> data/<asset>_historical_data.csv
  feature_calc.py      # engineer features + target -> data/<asset>_features.csv
  train_model.py       # train RandomForestClassifier -> models/trained_model_<asset>.pkl
  backtest.py          # simulate trading on held-out data -> results/backtest_results.png
run_pipeline.py         # runs the four scripts above in order
live_trader.py          # paper-trade live via Alpaca, using the trained model
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the offline pipeline

1. Edit `config.py`:
   - `ASSET_TYPE`: `"crypto"` (Kraken via ccxt) or `"stock"` (Yahoo Finance via yfinance)
   - `ASSET` / `SYMBOL`: e.g. `SPY` / `SPY`, or `BTC` / `BTC/USDT`
   - `TIMEFRAME`: `"1h"` or `"1d"` — **note:** this only controls the offline
     pipeline below; `live_trader.py` currently hardcodes hourly bars
     regardless of this setting (see [Known limitations](#known-limitations)).
2. Run everything:
   ```bash
   python3 run_pipeline.py
   ```
   Or run stages individually: `scripts/data_collector.py` →
   `scripts/feature_calc.py` → `scripts/train_model.py` → `scripts/backtest.py`.
3. Check `results/backtest_results.png` for the equity curve/drawdown chart,
   and the console output for accuracy, win rate, Sharpe, etc.

Training uses a fixed `random_state`, so re-running `train_model.py` on the
same feature file reproduces the same model.

## Live paper trading

`live_trader.py` connects to Alpaca's **paper** trading API, pulls recent
bars, recomputes features, loads the trained model for the asset configured
in `config.py`, and buys/sells based on the prediction.

Currently only wired up for **stocks** (Alpaca's equity bar/order endpoints).
Crypto isn't supported by `live_trader.py` yet even though `config.py` can be
set to a crypto asset for the offline pipeline.

### Credentials

Copy `.env.example` to `.env` and fill in a paper-trading key pair from your
[Alpaca dashboard](https://app.alpaca.markets):

```
ALPACA_API_KEY=your_alpaca_key_id
ALPACA_API_SECRET=your_alpaca_secret_key
```

`.env` is gitignored — keys are loaded via `python-dotenv` and never need to
be committed or typed into a shell command.

### Running it

```bash
# Interactive: choose single run or continuous (hourly) loop
python3 live_trader.py

# Non-interactive single cycle (for cron/schedulers)
python3 live_trader.py --once
```

Each cycle checks Alpaca's market clock first and skips (no API calls,
no trade) if the market is closed.

### Running it hourly via cron

Installed for this repo as:

```
0 * * * * cd /home/mhmor/trading-bot && venv/bin/python3 live_trader.py --once >> /home/mhmor/trading-bot/logs/live_trader.log 2>&1
```

Output (account info, predictions, trades, or "market is closed") accumulates
in `logs/live_trader.log` (gitignored). This is a local cron job — it only
fires while this machine (and WSL, if applicable) is running. On WSL2, either
keep a terminal/VS Code window attached, or set `vmIdleTimeout=-1` under
`[wsl2]` in `%UserProfile%\.wslconfig` to keep the VM running unattended.

There's no retry or alerting built in: a failed cycle (network blip, auth
error) just logs a traceback and waits for the next hourly fire.

## Known limitations

- **Live trader ignores `config.TIMEFRAME`** — it's hardcoded to hourly bars.
  Changing `config.py`'s timeframe only affects the offline pipeline.
- **No crypto support in `live_trader.py`** — only Alpaca's stock/equity API
  is implemented, even though the offline pipeline supports crypto via Kraken.
- **Feature calculation is duplicated** between `scripts/feature_calc.py` and
  `live_trader.py`'s `calculate_features()`. They must be kept in sync by
  hand — a change to one without the other will silently break predictions.
- **No real risk management** — live trading sizes buys as 95% of buying
  power, with no stop-loss, max position sizing, or partial-fill handling.
- **Model edge is unproven** — a single chronological 80/20 train/test split
  on ~600-2000 hours of data is small and high-variance. Backtest results
  have ranged from modestly beating buy-and-hold to roughly breaking even
  depending on asset/run. Don't take the backtest numbers as evidence of a
  real, durable trading edge.
- **Backtest/live fills differ** — the backtest assumes fills at each bar's
  close with a flat fee; live trading uses market orders at the latest trade
  price, so realized results will diverge from backtested ones.
