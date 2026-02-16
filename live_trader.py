"""
Live paper trading with Alpaca
Uses trained ML model to make real-time trading decisions
"""

import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import joblib
import time
from datetime import datetime, timedelta
import sys
from config import SYMBOL, MODEL_PATH

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
    
    def get_historical_data(self, days=30):
        """
        Fetch recent historical data for feature engineering
        
        Args:
            days: Number of days to fetch
        """
        print(f"\n{Colors.CYAN}Fetching {days} days of historical data...{Colors.END}")
        
        # Calculate start date
        end = datetime.now()
        start = end - timedelta(days=days)
        
        # Fetch bars (format dates as YYYY-MM-DD)
        barset = self.api.get_bars(
            self.symbol,
            tradeapi.TimeFrame.Hour,
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
        
        print(f"{Colors.GREEN}✓ Fetched {len(barset)} hourly bars{Colors.END}")
        return barset
    
    def calculate_features(self, df):
        """
        Calculate technical indicators (same as feature_calculator.py)
        """
        df = df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Price features
        df['returns_1h'] = df['close'].pct_change(1)
        df['returns_24h'] = df['close'].pct_change(24)
        
        # Moving averages
        df['sma_7'] = df['close'].rolling(window=7).mean()
        df['sma_25'] = df['close'].rolling(window=25).mean()
        df['sma_99'] = df['close'].rolling(window=99).mean()
        df['price_to_sma7'] = df['close'] / df['sma_7']
        df['price_to_sma25'] = df['close'] / df['sma_25']
        
        # EMAs
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # Volatility
        df['volatility_24h'] = df['close'].rolling(window=24).std()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # Volume
        df['volume_sma_24h'] = df['volume'].rolling(window=24).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_24h']
        
        # Time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        return df
    
    def make_prediction(self):
        """
        Get current market data and make a prediction
        """
        # Fetch historical data
        df = self.get_historical_data(days=30)
        
        # Calculate features
        df = self.calculate_features(df)
        
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
    
    def run_once(self):
        """Run one trading cycle"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
        print(f"Trading Cycle - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}{Colors.END}\n")
        
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
    # ⚠️ REPLACE THESE WITH YOUR ALPACA PAPER TRADING KEYS
    API_KEY = 'PKV3TUQI4PELB7AXLJGOZ3VTVF'
    API_SECRET = 'CdnbfnhGy2EXhYvqApLnjHTY9TLtfCEoVkg6JZVtj5Vq'

    
    trader = LiveTrader(
        api_key=API_KEY,
        api_secret=API_SECRET,
        symbol=SYMBOL,
        model_path=MODEL_PATH
    )
        
    # Run once or continuously
    print("\nChoose mode:")
    print("1. Run once (single prediction)")
    print("2. Run continuously (check every hour)")
    
    choice = input("\nEnter 1 or 2: ")
    
    if choice == '1':
        trader.run_once()
    else:
        trader.run_continuous(interval_minutes=60)