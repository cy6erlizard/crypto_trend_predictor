# scripts/chart_generation.py
import os
import pandas as pd
import plotly.graph_objects as go
from utils.config import RAW_DATA_DIR, CHARTS_DIR

def generate_candlestick_chart(df, output_path, title=""):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Candlesticks'
    )])
    
    fig.update_layout(title=title, yaxis_title='Price (USDT)')
    fig.write_image(output_path)

def main():
    os.makedirs(CHARTS_DIR, exist_ok=True)
    data_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')]
    
    for file in data_files:
        df = pd.read_csv(os.path.join(RAW_DATA_DIR, file), index_col='timestamp', parse_dates=True)
        symbol, interval = file.replace('.csv', '').split('_')
        chart_filename = f"{symbol}_{interval}.png"
        chart_path = os.path.join(CHARTS_DIR, chart_filename)
        generate_candlestick_chart(df, chart_path, title=f"{symbol} {interval} Chart")
        print(f"Chart saved to {chart_path}")

if __name__ == "__main__":
    main()
