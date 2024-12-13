# scripts/trading.py
import os
import pandas as pd
import joblib
from binance.client import Client
from utils.config import (BINANCE_API_KEY, BINANCE_API_SECRET, PREDICTIVE_MODEL_PATH,
                          TRADE_SYMBOL, TRADE_QUANTITY, STOP_LOSS_PERCENTAGE, TAKE_PROFIT_PERCENTAGE)
import logging

# Setup logging
logging.basicConfig(filename='trading.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

def load_model():
    model = joblib.load(PREDICTIVE_MODEL_PATH)
    return model

def get_latest_data():
    # Fetch latest data point
    df = pd.read_csv(os.path.join("data", "raw", "BTCUSDT_1h.csv"), index_col='timestamp', parse_dates=True)
    latest = df.iloc[-1]
    return latest

def generate_features(latest):
    # Simplified: In practice, ensure features match training
    features = {
        'RSI': latest['RSI'],
        'MACD': latest['MACD'],
        'SMA': latest['SMA'],
        # Add pattern dummies as needed
    }
    # Convert to DataFrame
    feature_df = pd.DataFrame([features])
    return feature_df

def place_order(client, side, quantity, symbol=TRADE_SYMBOL):
    try:
        order = client.create_order(
            symbol=symbol,
            side=side,
            type='MARKET',
            quantity=quantity
        )
        logging.info(f"Placed {side} order: {order}")
    except Exception as e:
        logging.error(f"Error placing order: {e}")

def main():
    model = load_model()
    latest_data = get_latest_data()
    features = generate_features(latest_data)
    
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]  # Probability of upward movement
    
    client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)
    
    if prediction[0] == 1 and probability > 0.7:
        # Buy Signal
        place_order(client, 'BUY', TRADE_QUANTITY)
        logging.info("Buy signal generated.")
    elif prediction[0] == 0 and (1 - probability) > 0.7:
        # Sell Signal
        place_order(client, 'SELL', TRADE_QUANTITY)
        logging.info("Sell signal generated.")
    else:
        logging.info("No trading signal generated.")

if __name__ == "__main__":
    main()
