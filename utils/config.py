# utils/config.py
import os

# Data directories
RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")
CHARTS_DIR = os.path.join("data", "charts")

# Model directories
OBJECT_DETECTION_MODEL_DIR = os.path.join("models", "object_detection")
PREDICTIVE_MODEL_DIR = os.path.join("models", "predictive_model")

# API Keys (Ensure to set these securely)
BINANCE_API_KEY = "your_binance_api_key"
BINANCE_API_SECRET = "your_binance_api_secret"

# Trading Parameters
TRADE_SYMBOL = "BTCUSDT"
TRADE_QUANTITY = 0.001
STOP_LOSS_PERCENTAGE = 0.02  # 2%
TAKE_PROFIT_PERCENTAGE = 0.04  # 4%

# YOLO Configuration
YOLO_MODEL_PATH = os.path.join(OBJECT_DETECTION_MODEL_DIR, "best.pt")
YOLO_CLASSES = [
    "head_and_shoulders", "inverse_head_and_shoulders", "double_top", "double_bottom",
    "ascending_triangle", "descending_triangle", "symmetrical_triangle", "flag", "pennant",
    "channel", "wedge"
]

# Predictive Model Configuration
PREDICTIVE_MODEL_PATH = os.path.join(PREDICTIVE_MODEL_DIR, "xgboost_model.json")

