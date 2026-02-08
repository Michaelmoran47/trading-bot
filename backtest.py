import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Color codes for output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

class Backtester:
    def __init__(self, initial_capital=10000, fee_rate=0.001):
        """
        Initialize backtester
        
        Args:
            initial_capital: Starting cash in USD
            fee_rate: Trading fee as decimal (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        
        # Portfolio state
        self.cash = initial_capital
        self.btc_holdings = 0
        self.portfolio_value = initial_capital
        
        # Trade history
        self.trades = []
        self.portfolio_history = []
        
    def load_model(self, model_path='trained_model.pkl'):
        """Load the trained ML model"""
        print(f"Loading model from {model_path}...")
        self.model = joblib.load(model_path)
        print(f"{Colors.GREEN}✓ Model loaded{Colors.END}")
        
    def load_data(self, data_path='btc_features.csv'):
        """Load the feature data"""
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Drop NaN rows
        df = df.dropna()
        
        print(f"Loaded {len(df)} records")
        return df
    
    def prepare_features(self, df):
        """Prepare features for prediction (same as training)"""
        # Keep timestamp and close for reference
        timestamps = df['timestamp']
        prices = df['close']
        
        # Drop non-feature columns
        feature_df = df.drop(['timestamp', 'target'], axis=1, errors='ignore')
        
        return feature_df, timestamps, prices
    
    def execute_trade(self, action, price, timestamp):
        """
        Execute a trade
        
        Args:
            action: 'buy' or 'sell'
            price: Current BTC price
            timestamp: Time of trade
        """
        if action == 'buy' and self.cash > 0:
            # Buy BTC with all available cash
            fee = self.cash * self.fee_rate
            btc_bought = (self.cash - fee) / price
            
            self.btc_holdings += btc_bought
            self.cash = 0
            
            self.trades.append({
                'timestamp': timestamp,
                'action': 'BUY',
                'price': price,
                'btc_amount': btc_bought,
                'fee': fee,
                'portfolio_value': self.btc_holdings * price
            })
            
        elif action == 'sell' and self.btc_holdings > 0:
            # Sell all BTC for cash
            cash_received = self.btc_holdings * price
            fee = cash_received * self.fee_rate
            
            self.cash = cash_received - fee
            
            self.trades.append({
                'timestamp': timestamp,
                'action': 'SELL',
                'price': price,
                'btc_amount': self.btc_holdings,
                'fee': fee,
                'portfolio_value': self.cash
            })
            
            self.btc_holdings = 0
    
    def run_backtest(self, df, test_split=0.8):
        """
        Run the backtest on test data
        
        Args:
            df: DataFrame with features
            test_split: Use data after this proportion for testing
        """
        print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}")
        print("RUNNING BACKTEST")
        print(f"{'='*60}{Colors.END}\n")
        
        # Split data (same as training)
        split_index = int(len(df) * test_split)
        test_df = df.iloc[split_index:].copy().reset_index(drop=True)
        
        print(f"Backtesting on {len(test_df)} hours of data")
        print(f"Initial capital: ${self.initial_capital:,.2f}")
        print(f"Trading fee: {self.fee_rate*100}%\n")
        
        # Prepare features
        X_test, timestamps, prices = self.prepare_features(test_df)
        
        # Get predictions
        print("Generating predictions...")
        predictions = self.model.predict(X_test)
        
        # Track current position
        position = None  # None, 'long', or 'flat'
        
        # Simulate trading
        for i in range(len(predictions)):
            current_price = prices.iloc[i]
            current_time = timestamps.iloc[i]
            prediction = predictions[i]
            
            # Trading logic: Simple strategy
            # If model predicts UP (1) and we're not in a position, BUY
            # If model predicts DOWN (0) and we're in a position, SELL
            
            if prediction == 1 and position != 'long':
                # Prediction: price will go up, so BUY
                self.execute_trade('buy', current_price, current_time)
                position = 'long'
                
            elif prediction == 0 and position == 'long':
                # Prediction: price will go down, so SELL
                self.execute_trade('sell', current_price, current_time)
                position = 'flat'
            
            # Calculate portfolio value at this point
            if self.btc_holdings > 0:
                portfolio_value = self.btc_holdings * current_price
            else:
                portfolio_value = self.cash
            
            self.portfolio_history.append({
                'timestamp': current_time,
                'portfolio_value': portfolio_value,
                'price': current_price,
                'position': position
            })
        
        # Close any open position at the end
        if self.btc_holdings > 0:
            final_price = prices.iloc[-1]
            final_time = timestamps.iloc[-1]
            self.execute_trade('sell', final_price, final_time)
        
        print(f"{Colors.GREEN}✓ Backtest complete{Colors.END}")
        
    def calculate_metrics(self):
        """Calculate performance metrics"""
        print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}")
        print("BACKTEST RESULTS")
        print(f"{'='*60}{Colors.END}\n")
        
        final_value = self.cash
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        # Calculate buy and hold return
        portfolio_df = pd.DataFrame(self.portfolio_history)
        buy_hold_return = (portfolio_df['price'].iloc[-1] - portfolio_df['price'].iloc[0]) / portfolio_df['price'].iloc[0] * 100
        
        # Number of trades
        num_trades = len(self.trades)
        
        # Calculate total fees paid
        total_fees = sum([trade['fee'] for trade in self.trades])
        
        # Win rate
        winning_trades = 0
        for i in range(0, len(self.trades)-1, 2):  # Buy-sell pairs
            if i+1 < len(self.trades):
                buy_price = self.trades[i]['price']
                sell_price = self.trades[i+1]['price']
                if sell_price > buy_price:
                    winning_trades += 1
        
        total_trade_pairs = num_trades // 2
        win_rate = (winning_trades / total_trade_pairs * 100) if total_trade_pairs > 0 else 0
        
        # Print results
        print(f"Initial Capital:     ${self.initial_capital:,.2f}")
        print(f"Final Value:         ${final_value:,.2f}")
        
        # Color code the return
        if total_return > 0:
            return_color = Colors.GREEN
        else:
            return_color = Colors.RED
        
        print(f"Total Return:        {return_color}{total_return:+.2f}%{Colors.END}")
        print(f"Buy & Hold Return:   {buy_hold_return:+.2f}%")
        
        if total_return > buy_hold_return:
            print(f"{Colors.GREEN}✓ Strategy outperformed buy & hold!{Colors.END}")
        else:
            print(f"{Colors.YELLOW}⚠ Strategy underperformed buy & hold{Colors.END}")
        
        print(f"\nNumber of Trades:    {num_trades}")
        print(f"Trade Pairs:         {total_trade_pairs}")
        print(f"Win Rate:            {win_rate:.1f}%")
        print(f"Total Fees Paid:     ${total_fees:,.2f}")
        
        # Calculate max drawdown
        portfolio_df['peak'] = portfolio_df['portfolio_value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['peak']) / portfolio_df['peak'] * 100
        max_drawdown = portfolio_df['drawdown'].min()
        
        print(f"Max Drawdown:        {max_drawdown:.2f}%")
        
        # Sharpe ratio (simplified - assumes risk-free rate of 0)
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
        sharpe = portfolio_df['returns'].mean() / portfolio_df['returns'].std() * np.sqrt(24*365) if portfolio_df['returns'].std() != 0 else 0
        
        print(f"Sharpe Ratio:        {sharpe:.2f}")
        
        return portfolio_df
    
    def plot_results(self, portfolio_df):
        """Plot backtest results"""
        print(f"\n{Colors.CYAN}Generating plot...{Colors.END}")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Plot 1: Portfolio value vs BTC price
        ax1_twin = ax1.twinx()
        
        ax1.plot(portfolio_df['timestamp'], portfolio_df['portfolio_value'], 
                 label='Portfolio Value', color='blue', linewidth=2)
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', 
                    label='Initial Capital', alpha=0.7)
        
        ax1_twin.plot(portfolio_df['timestamp'], portfolio_df['price'], 
                      label='BTC Price', color='orange', alpha=0.6)
        
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1_twin.set_ylabel('BTC Price ($)', fontsize=12)
        ax1.set_title('Backtest Results: Portfolio Value vs BTC Price', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Drawdown
        ax2.fill_between(portfolio_df['timestamp'], portfolio_df['drawdown'], 0, 
                         color='red', alpha=0.3, label='Drawdown')
        ax2.plot(portfolio_df['timestamp'], portfolio_df['drawdown'], 
                 color='darkred', linewidth=1)
        
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_title('Portfolio Drawdown Over Time', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig('backtest_results.png', dpi=150, bbox_inches='tight')
        print(f"{Colors.GREEN}✓ Plot saved to backtest_results.png{Colors.END}")
        
        # Show plot (optional - comment out if running headless)
        # plt.show()

if __name__ == "__main__":
    # Initialize backtester
    backtester = Backtester(
        initial_capital=10000,  # Start with $10,000
        fee_rate=0.001          # 0.1% trading fee (typical for crypto)
    )
    
    # Load model and data
    backtester.load_model('trained_model.pkl')
    df = backtester.load_data('btc_features.csv')
    
    # Run backtest
    backtester.run_backtest(df, test_split=0.8)
    
    # Calculate and display metrics
    portfolio_df = backtester.calculate_metrics()
    
    # Plot results
    backtester.plot_results(portfolio_df)
    
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}")
    print("BACKTEST COMPLETE")
    print(f"{'='*60}{Colors.END}\n")