import ccxt
import pandas as pd
from datetime import datetime
import time

def fetch_all_ohlcv_data(symbol='BTC/USDT', timeframe='1h', since=None):
    """
    Fetch all available OHLCV data from Kraken
    """
    exchange = ccxt.kraken()
    
    all_data = []
    limit = 720  # Kraken's max per request
    
    while True:
        print(f"Fetching data since {since}...")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        
        if len(ohlcv) == 0:
            break
            
        all_data.extend(ohlcv)
        
        # Get timestamp of last candle for next request
        since = ohlcv[-1][0] + 1
        
        # Be nice to the API
        time.sleep(exchange.rateLimit / 1000)
        
        print(f"Fetched {len(ohlcv)} candles, total: {len(all_data)}")
        
        # Stop if we got less than requested (means we're at the end)
        if len(ohlcv) < limit:
            break
    
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset=['timestamp'])
    
    return df

if __name__ == "__main__":
    # Fetch from 1 year ago to now
    one_year_ago = int((datetime.now().timestamp() - 365*24*60*60) * 1000)
    
    print("Fetching 1 year of BTC hourly data...")
    data = fetch_all_ohlcv_data(since=one_year_ago)
    
    data.to_csv('btc_historical_data.csv', index=False)
    print(f"\nSaved {len(data)} records")
    print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    print(data.head())