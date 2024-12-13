# scripts/data_collection.py
import os
import pandas as pd
from binance.client import Client
from utils.config import RAW_DATA_DIR, BINANCE_API_KEY, BINANCE_API_SECRET

def fetch_historical_data(symbol="BTCUSDT", interval="1h", start_str="1 Jan 2020", end_str=None):
    client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)
    
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
               'close_time', 'quote_asset_volume', 'number_of_trades',
               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    
    df = pd.DataFrame(klines, columns=columns)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    
    return df

def save_data(df, symbol="BTCUSDT", interval="1h"):
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    file_path = os.path.join(RAW_DATA_DIR, f"{symbol}_{interval}.csv")
    df.to_csv(file_path)
    print(f"Data saved to {file_path}")

if __name__ == "__main__":
    df = fetch_historical_data()
    save_data(df)
