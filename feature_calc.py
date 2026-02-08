import pandas as pd
import numpy as np

def add_technical_features(df):
    """
    Add technical indicator features to the dataframe
    df should have columns: timestamp, open, high, low, close, volume
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Sort by timestamp to ensure correct calculations
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # ===== PRICE-BASED FEATURES =====
    
    # Returns (percent change)
    df['returns_1h'] = df['close'].pct_change(1)
    df['returns_24h'] = df['close'].pct_change(24)
    
    # Simple Moving Averages
    df['sma_7'] = df['close'].rolling(window=7).mean()
    df['sma_25'] = df['close'].rolling(window=25).mean()
    df['sma_99'] = df['close'].rolling(window=99).mean()
    
    # Price position relative to SMAs
    df['price_to_sma7'] = df['close'] / df['sma_7']
    df['price_to_sma25'] = df['close'] / df['sma_25']
    
    # Exponential Moving Averages
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    
    # ===== VOLATILITY FEATURES =====
    
    # Standard deviation (volatility)
    df['volatility_24h'] = df['close'].rolling(window=24).std()
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # ===== MOMENTUM FEATURES =====
    
    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    # ===== VOLUME FEATURES =====
    
    df['volume_sma_24h'] = df['volume'].rolling(window=24).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_24h']
    
    # ===== TIME FEATURES =====
    
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    return df

def create_target(df, horizon=1):
    """
    Create target variable: 1 if price goes up in next 'horizon' hours, 0 otherwise
    """
    df = df.copy()
    df['target'] = (df['close'].shift(-horizon) > df['close']).astype(int)
    return df

if __name__ == "__main__":
    # Load the data
    print("Loading data...")
    df = pd.read_csv('btc_historical_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"Loaded {len(df)} records")
    
    # Add features
    print("Calculating features...")
    df = add_technical_features(df)
    
    # Create target (predict if price goes up in next hour)
    df = create_target(df, horizon=1)
    
    # Drop rows with NaN (from rolling calculations at the start)
    df = df.dropna()
    
    print(f"After feature engineering: {len(df)} records with {len(df.columns)} columns")
    
    # Save
    df.to_csv('btc_features.csv', index=False)
    print("Saved features to btc_features.csv")
    
    # Show some stats
    print(f"\nTarget distribution:")
    print(df['target'].value_counts())
    print(f"\nFeature columns: {list(df.columns)}")