# scripts/predictive_modeling.py
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from utils.config import CHARTS_DIR, PREDICTIVE_MODEL_PATH
import ta

def load_data():
    detected_patterns = pd.read_csv(os.path.join(CHARTS_DIR, 'detected_patterns.csv'))
    confirmed_trends = pd.read_csv(os.path.join(CHARTS_DIR, 'confirmed_trends.csv'))
    
    # Load price data
    price_data = pd.read_csv(os.path.join("data", "raw", "BTCUSDT_1h.csv"), index_col='timestamp', parse_dates=True)
    
    # Merge patterns with price data
    # Simplified: Assuming each chart corresponds to a timestamp
    # You need to adjust based on your actual data mapping
    data = price_data.join(detected_patterns.set_index('chart'), how='left')
    data.fillna(method='ffill', inplace=True)
    
    # Feature Engineering
    data['RSI'] = ta.momentum.RSIIndicator(data['close'], window=14).rsi()
    data['MACD'] = ta.trend.MACD(data['close']).macd()
    data['SMA'] = ta.trend.SMAIndicator(data['close'], window=30).sma_indicator()
    
    # Encode patterns
    pattern_dummies = pd.get_dummies(data['pattern'])
    data = pd.concat([data, pattern_dummies], axis=1)
    
    # Define target: Next period price movement
    data['future_close'] = data['close'].shift(-1)
    data['price_movement'] = np.where(data['future_close'] > data['close'], 1, 0)
    data.dropna(inplace=True)
    
    return data

def train_model(data):
    feature_cols = list(data.columns)
    feature_cols.remove('close')
    feature_cols.remove('future_close')
    feature_cols.remove('price_movement')
    
    X = data[feature_cols]
    y = data['price_movement']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_val, y_val)], verbose=True)
    
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    
    # Save the model
    joblib.dump(model, PREDICTIVE_MODEL_PATH)
    print(f"Model saved to {PREDICTIVE_MODEL_PATH}")

def main():
    data = load_data()
    train_model(data)

if __name__ == "__main__":
    main()
