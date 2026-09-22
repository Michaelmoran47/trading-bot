"""
Live paper trading with Alpaca
Uses trained ML model to make real-time trading decisions
"""

import alpaca_trade_api as tradeapi
import os
import joblib
import time
from datetime import datetime, timedelta
import sys
from dotenv import load_dotenv
from config import SYMBOL, MODEL_PATH, TIMEFRAME
from scripts.feature_calc import add_technical_features

load_dotenv()

# Maps config.TIMEFRAME to the matching Alpaca bar timeframe, and how many
# calendar days to fetch to cover the longest rolling feature window (99
# periods) with room for weekends/holidays.
TIMEFRAME_MAP = {
    "1h": (tradeapi.TimeFrame.Hour, 30),
    "1d": (tradeapi.TimeFrame.Day, 200),
}

# Color codes
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

class LiveTrader:
    def __init__(self, api_key, api_secret, symbol='AAPL', model_path='models/trained_model.pkl'):
        """
        Initialize live trader
        
        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            symbol: Stock symbol to trade
            model_path: Path to trained model
        """
        # Initialize Alpaca API (paper trading)
        self.api = tradeapi.REST(
            api_key,
            api_secret,
            base_url='https://paper-api.alpaca.markets'  # Paper trading endpoint
        )

        self.symbol = symbol
        self.position = None  # Track if we're in a position

        if TIMEFRAME not in TIMEFRAME_MAP:
            raise ValueError(
                f'config.TIMEFRAME must be "1h" or "1d", got: {TIMEFRAME!r}'
            )
        self.bar_timeframe, self.lookback_days = TIMEFRAME_MAP[TIMEFRAME]
        
        # Load trained model
        print(f"{Colors.CYAN}Loading model from {model_path}...{Colors.END}")
        self.model = joblib.load(model_path)
        print(f"{Colors.GREEN}✓ Model loaded{Colors.END}")
        
    def get_account_info(self):
        """Get account information"""
        account = self.api.get_account()
        print(f"\n{Colors.BOLD}Account Info:{Colors.END}")
        print(f"  Cash: ${float(account.cash):,.2f}")
        print(f"  Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"  Buying Power: ${float(account.buying_power):,.2f}")
        return account
    
    def get_current_position(self):
        """Check if we have an open position"""
        try:
            position = self.api.get_position(self.symbol)
            qty = int(position.qty)
            print(f"{Colors.GREEN}Current position: {qty} shares of {self.symbol}{Colors.END}")
            return qty
        except:
            print(f"{Colors.YELLOW}No position in {self.symbol}{Colors.END}")
            return 0
    
    def get_historical_data(self, days=None):
        """
        Fetch recent historical data for feature engineering

        Args:
            days: Number of days to fetch (defaults to enough for the
                  configured timeframe's longest rolling feature window)
        """
        if days is None:
            days = self.lookback_days
        print(f"\n{Colors.CYAN}Fetching {days} days of historical data...{Colors.END}")

        # Calculate start date
        end = datetime.now()
        start = end - timedelta(days=days)

        # Fetch bars (format dates as YYYY-MM-DD)
        barset = self.api.get_bars(
            self.symbol,
            self.bar_timeframe,
            start=start.strftime('%Y-%m-%d'),
            end=end.strftime('%Y-%m-%d'),
            feed='iex',
        ).df

        # Rename columns to match our format
        barset = barset.reset_index()
        barset = barset.rename(columns={
            'timestamp': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })

        print(f"{Colors.GREEN}✓ Fetched {len(barset)} bars{Colors.END}")
        return barset

    def make_prediction(self):
        """
        Get current market data and make a prediction
        """
        # Fetch historical data
        df = self.get_historical_data()

        # Calculate features (same logic used to build the training data)
        df = add_technical_features(df)
        
        # Get most recent complete row
        df = df.dropna()
        latest = df.iloc[-1:]

        # Drop non-feature columns (including extra Alpaca columns)
        columns_to_drop = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
        features = latest.drop([col for col in columns_to_drop if col in latest.columns], axis=1)
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        print(f"\n{Colors.BOLD}Prediction:{Colors.END}")
        print(f"  Direction: {'UP (1)' if prediction == 1 else 'DOWN (0)'}")
        print(f"  Confidence: {probability[prediction]*100:.1f}%")
        
        return prediction, probability[prediction]
    
    def execute_trade(self, signal):
        """
        Execute trade based on signal
        
        Args:
            signal: 1 (buy) or 0 (sell/hold cash)
        """
        current_position = self.get_current_position()
        
        # Get current price
        latest_trade = self.api.get_latest_trade(self.symbol, feed='iex')
        current_price = latest_trade.price
        print(f"Current price: ${current_price:.2f}")
        
        if signal == 1 and current_position == 0:
            # BUY signal and we're not in a position
            account = self.api.get_account()
            buying_power = float(account.buying_power)
            
            # Calculate shares to buy (use 95% of buying power for safety)
            shares_to_buy = int((buying_power * 0.95) / current_price)
            
            if shares_to_buy > 0:
                print(f"\n{Colors.GREEN}{Colors.BOLD}🔥 BUYING {shares_to_buy} shares of {self.symbol}{Colors.END}")
                
                # Place market order
                order = self.api.submit_order(
                    symbol=self.symbol,
                    qty=shares_to_buy,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
                
                print(f"{Colors.GREEN}✓ Order placed: {order.id}{Colors.END}")
            else:
                print(f"{Colors.YELLOW}⚠ Not enough buying power{Colors.END}")
                
        elif signal == 0 and current_position > 0:
            # SELL signal and we have a position
            print(f"\n{Colors.RED}{Colors.BOLD}📉 SELLING {current_position} shares of {self.symbol}{Colors.END}")
            
            # Place market order to close position
            order = self.api.submit_order(
                symbol=self.symbol,
                qty=current_position,
                side='sell',
                type='market',
                time_in_force='day'
            )
            
            print(f"{Colors.GREEN}✓ Order placed: {order.id}{Colors.END}")
            
        else:
            print(f"{Colors.CYAN}↔ No action needed{Colors.END}")
    
    def is_market_open(self):
        """Check whether the exchange is currently open for trading"""
        clock = self.api.get_clock()
        return clock.is_open

    def run_once(self):
        """Run one trading cycle"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
        print(f"Trading Cycle - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}{Colors.END}\n")

        if not self.is_market_open():
            print(f"{Colors.YELLOW}Market is closed — skipping this cycle{Colors.END}")
            return

        # Get account info
        self.get_account_info()
        
        # Make prediction
        signal, confidence = self.make_prediction()
        
        # Execute trade
        self.execute_trade(signal)
        
    def run_continuous(self, interval_minutes=60):
        """
        Run trading loop continuously
        
        Args:
            interval_minutes: How often to check and trade (default: every hour)
        """
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
        print(f"LIVE PAPER TRADING STARTED")
        print(f"{'='*60}{Colors.END}")
        print(f"Symbol: {self.symbol}")
        print(f"Check interval: Every {interval_minutes} minutes")
        print(f"Press Ctrl+C to stop\n")
        
        try:
            while True:
                self.run_once()
                
                # Wait for next cycle
                print(f"\n{Colors.YELLOW}💤 Sleeping for {interval_minutes} minutes...{Colors.END}")
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}Trading stopped by user{Colors.END}")
            self.get_account_info()

if __name__ == "__main__":
    API_KEY = os.getenv('ALPACA_API_KEY')
    API_SECRET = os.getenv('ALPACA_API_SECRET')
    if not API_KEY or not API_SECRET:
        raise RuntimeError('Set ALPACA_API_KEY and ALPACA_API_SECRET before starting the trader')


    trader = LiveTrader(
        api_key=API_KEY,
        api_secret=API_SECRET,
        symbol=SYMBOL,
        model_path=MODEL_PATH
    )
        
    if '--once' in sys.argv:
        # Non-interactive single cycle, e.g. for a cron/scheduled invocation.
        trader.run_once()
    else:
        # Run once or continuously
        print("\nChoose mode:")
        print("1. Run once (single prediction)")
        print("2. Run continuously (check every hour)")

        choice = input("\nEnter 1 or 2: ")

        if choice == '1':
            trader.run_once()
        else:
            trader.run_continuous(interval_minutes=60)