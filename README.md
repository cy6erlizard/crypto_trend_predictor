# Crypto Trend Predictor

![Crypto Trend Predictor](https://github.com/yourusername/crypto_trend_predictor/blob/main/assets/banner.png)

**Crypto Trend Predictor** is an advanced Python-based project that leverages object detection and machine learning to identify chart patterns in cryptocurrency markets, analyze trend sequences, and predict future price movements. By automating the detection of recognizable chart patterns and utilizing these patterns to forecast market trends, this tool aims to assist traders in making informed and profitable trading decisions.

---

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage](#usage)
    - [1. Data Collection](#1-data-collection)
    - [2. Chart Generation](#2-chart-generation)
    - [3. Annotation](#3-annotation)
    - [4. Train Object Detection Model](#4-train-object-detection-model)
    - [5. Pattern Sequence Analysis](#5-pattern-sequence-analysis)
    - [6. Predictive Modeling](#6-predictive-modeling)
    - [7. Execute Trading](#7-execute-trading)
6. [Prerequisites](#prerequisites)
7. [Dependencies](#dependencies)
8. [Contributing](#contributing)
9. [License](#license)
10. [Contact](#contact)
11. [Acknowledgments](#acknowledgments)

---

## Features

- **Automated Data Collection:** Fetches historical cryptocurrency price data from Binance API.
- **Chart Generation:** Converts OHLC (Open, High, Low, Close) data into candlestick charts.
- **Object Detection:** Utilizes YOLOv8 to detect predefined chart patterns within candlestick charts.
- **Pattern Sequence Analysis:** Analyzes sequences of detected patterns to confirm market trends.
- **Predictive Modeling:** Employs XGBoost to forecast future price movements based on detected patterns and technical indicators.
- **Automated Trading:** Integrates with Binance API to execute buy/sell orders based on model predictions.
- **Modular Design:** Structured into distinct modules for easy maintenance and scalability.
- **Logging and Monitoring:** Comprehensive logging for tracking system performance and trading activities.

---

## Project Structure

```plaintext
crypto_trend_predictor/
├── data/
│   ├── raw/
│   ├── processed/
│   └── charts/
├── models/
│   ├── object_detection/
│   │   └── data/
│   │       ├── train/
│   │       └── val/
│   └── predictive_model/
├── scripts/
│   ├── data_collection.py
│   ├── chart_generation.py
│   ├── annotation.py
│   ├── train_object_detection.py
│   ├── pattern_sequence_analysis.py
│   ├── predictive_modeling.py
│   └── trading.py
├── utils/
│   ├── helpers.py
│   └── config.py
├── requirements.txt
├── main.py
├── README.md
└── assets/
    └── banner.png
```

- **data/**: Stores all data-related files.
  - **raw/**: Contains raw OHLC data fetched from Binance.
  - **processed/**: Holds cleaned and processed data ready for analysis.
  - **charts/**: Stores generated candlestick chart images.

- **models/**: Stores trained models.
  - **object_detection/**: Contains the YOLOv8 object detection model and related data.
    - **data/train/**: Training dataset for object detection.
    - **data/val/**: Validation dataset for object detection.
  - **predictive_model/**: Stores the trained XGBoost predictive model.

- **scripts/**: Contains all Python scripts for different stages of the pipeline.
  - **data_collection.py**: Script to fetch historical data from Binance.
  - **chart_generation.py**: Generates candlestick charts from OHLC data.
  - **annotation.py**: Handles annotation conversion for object detection.
  - **train_object_detection.py**: Trains the YOLOv8 object detection model.
  - **pattern_sequence_analysis.py**: Analyzes detected patterns to confirm trends.
  - **predictive_modeling.py**: Trains the XGBoost model for price prediction.
  - **trading.py**: Executes trading strategies based on model predictions.

- **utils/**: Contains utility scripts and configuration files.
  - **helpers.py**: Helper functions (currently placeholder).
  - **config.py**: Central configuration file for project settings.

- **requirements.txt**: Lists all Python dependencies required for the project.

- **main.py**: Orchestrates the execution of different scripts based on user input.

- **assets/**: Contains asset files like banners or diagrams.

---

## Installation

Follow the steps below to set up the **Crypto Trend Predictor** project on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/crypto_trend_predictor.git
cd crypto_trend_predictor
```

### 2. Set Up a Virtual Environment (Recommended)

It's advisable to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

Ensure you have `pip` installed and run:

```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys

The project interacts with the Binance API for data collection and trading. You need to obtain API keys from Binance.

1. **Obtain API Keys:**
   - Sign up/log in to your [Binance account](https://www.binance.com/).
   - Navigate to the API Management section.
   - Create a new API key and secret.

2. **Configure API Keys:**
   - Open `utils/config.py`.
   - Replace the placeholder strings with your actual Binance API credentials.

   ```python
   # utils/config.py

   BINANCE_API_KEY = "your_binance_api_key"
   BINANCE_API_SECRET = "your_binance_api_secret"
   ```

   **Security Note:** Never expose your API keys in public repositories. Consider using environment variables or secret management tools for enhanced security.

---

## Configuration

All configurable parameters are centralized in `utils/config.py`. Below is an overview of the key configurations:

```python
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
```

**Key Parameters:**

- **Data Directories:** Paths for raw, processed data, and generated charts.
- **Model Directories:** Paths to store trained object detection and predictive models.
- **API Keys:** Binance API credentials for data access and trading.
- **Trading Parameters:** Define trading symbol, quantity, stop loss, and take profit percentages.
- **YOLO Configuration:** Path to the trained YOLOv8 model and the list of chart pattern classes.
- **Predictive Model Configuration:** Path to the trained XGBoost model.

---

## Usage

The project is modular, with each stage of the pipeline handled by a specific script. You can orchestrate the entire workflow using `main.py` by specifying the desired step. Below are detailed instructions for each step.

### 1. Data Collection

**Script:** `scripts/data_collection.py`

**Description:** Fetches historical OHLC (Open, High, Low, Close) data from Binance API and saves it to the `data/raw/` directory.

**Run the Script:**

```bash
python main.py --step collect_data
```

**Customization:**

- To fetch data for different symbols or intervals, modify the parameters in `data_collection.py` or enhance the script to accept command-line arguments.

**Output:**

- Saves CSV files named as `{symbol}_{interval}.csv` in `data/raw/`.

### 2. Chart Generation

**Script:** `scripts/chart_generation.py`

**Description:** Converts raw OHLC data into candlestick chart images and saves them to `data/charts/`.

**Run the Script:**

```bash
python main.py --step generate_charts
```

**Output:**

- Saves chart images as `{symbol}_{interval}.png` in `data/charts/`.

**Dependencies:**

- [Plotly](https://plotly.com/python/candlestick-charts/) for generating interactive candlestick charts.

### 3. Annotation

**Script:** `scripts/annotation.py`

**Description:** Converts annotation files from JSON (produced by annotation tools like LabelImg) to YOLO format.

**Manual Annotation Steps:**

1. **Install LabelImg:**

   ```bash
   pip install labelImg
   labelImg
   ```

2. **Annotate Charts:**
   - Open LabelImg.
   - Load images from `data/charts/`.
   - Draw bounding boxes around identified chart patterns.
   - Label each box with the appropriate pattern class from `YOLO_CLASSES`.
   - Save annotations in YOLO format.

**Run the Conversion Script:**

```bash
python main.py --step convert_annotations
```

**Output:**

- Saves YOLO-formatted annotation files (`.txt`) in `data/charts/yolo_annotations/`.

**Note:**

- Ensure that annotations are saved in the `data/charts/annotations/` directory before running the conversion script.

### 4. Train Object Detection Model

**Script:** `scripts/train_object_detection.py`

**Description:** Trains the YOLOv8 object detection model using the annotated chart images.

**Run the Script:**

```bash
python main.py --step train_yolo
```

**Requirements:**

- GPU with CUDA support is recommended for faster training.

**Output:**

- Saves the trained YOLOv8 model (`best.pt`) in `models/object_detection/`.

**Customization:**

- Adjust training parameters (epochs, batch size, image size) in `train_object_detection.py` as needed.

### 5. Pattern Sequence Analysis

**Script:** `scripts/pattern_sequence_analysis.py`

**Description:** Detects patterns in chart images using the trained YOLOv8 model, analyzes sequences of three consecutive patterns to confirm market trends, and saves the confirmed trends.

**Run the Script:**

```bash
python main.py --step detect_patterns
```

**Output:**

- `detected_patterns.csv`: Lists all detected patterns with associated chart images.
- `confirmed_trends.csv`: Lists confirmed bullish or bearish trends based on pattern sequences.

**Customization:**

- Define additional bullish or bearish sequences in `pattern_sequence_analysis.py` as per your trading strategy.

### 6. Predictive Modeling

**Script:** `scripts/predictive_modeling.py`

**Description:** Trains an XGBoost classifier to predict future price movements based on detected patterns and technical indicators.

**Run the Script:**

```bash
python main.py --step train_predictive_model
```

**Output:**

- Saves the trained XGBoost model (`xgboost_model.json`) in `models/predictive_model/`.

**Customization:**

- Modify feature engineering steps or model parameters in `predictive_modeling.py` to enhance performance.

### 7. Execute Trading

**Script:** `scripts/trading.py`

**Description:** Executes buy or sell orders on Binance based on the predictive model's signals.

**Run the Script:**

```bash
python main.py --step execute_trading
```

**Prerequisites:**

- Ensure that the predictive model (`xgboost_model.json`) is trained and available.
- API keys must have trading permissions enabled.
- **Caution:** Trading with real money involves significant risk. It's recommended to test with a paper trading account or small amounts initially.

**Output:**

- Logs trading activities in `trading.log`.

**Customization:**

- Adjust trading parameters (symbol, quantity, stop loss, take profit) in `utils/config.py` as per your risk tolerance and strategy.

---

## Prerequisites

- **Python 3.7 or higher**: Ensure that Python is installed on your system.
- **Git**: For version control and cloning the repository.
- **Binance Account**: To access Binance API for data collection and trading.
- **GPU (Optional but Recommended)**: For faster training of deep learning models.

---

## Dependencies

All project dependencies are listed in `requirements.txt`. Key libraries and frameworks include:

- **Data Handling:** `numpy`, `pandas`
- **Visualization:** `matplotlib`, `plotly`
- **API Interaction:** `python-binance`, `requests`
- **Technical Indicators:** `ta`
- **Computer Vision:** `opencv-python`, `ultralytics` (YOLOv8)
- **Machine Learning:** `torch`, `torchvision`, `scikit-learn`, `xgboost`
- **Web Framework:** `fastapi`, `uvicorn`
- **Utilities:** `PyYAML`, `joblib`, `labelImg`

**Installation:**

```bash
pip install -r requirements.txt
```

---

## Contributing

Contributions are welcome! Whether it's improving documentation, adding new features, or fixing bugs, your input is valuable.

### Steps to Contribute

1. **Fork the Repository**

   Click the "Fork" button at the top-right corner of this repository's page to create a personal copy.

2. **Clone Your Fork**

   ```bash
   git clone https://github.com/cy6erlizard/crypto_trend_predictor.git
   cd crypto_trend_predictor
   ```

3. **Create a New Branch**

   ```bash
   git checkout -b feature/YourFeatureName
   ```

4. **Make Your Changes**

   Implement your feature or fix the bug.

5. **Commit Your Changes**

   ```bash
   git add .
   git commit -m "Add feature XYZ"
   ```

6. **Push to Your Fork**

   ```bash
   git push origin feature/YourFeatureName
   ```

7. **Create a Pull Request**

   Navigate to the original repository and create a pull request from your fork's feature branch.

### Guidelines

- **Code Quality:** Ensure that your code adheres to PEP 8 standards. Use meaningful variable names and include comments where necessary.
- **Documentation:** Update the README and other relevant documentation to reflect your changes.
- **Testing:** If applicable, add tests to verify your changes.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For any inquiries or support, please contact:

- **Name:** Your Name
- **Email:** your.email@example.com
- **GitHub:** [yourusername](https://github.com/cy6erlizard)

---

## Acknowledgments

- **[Binance](https://www.binance.com/):** For providing robust APIs for cryptocurrency data and trading.
- **[Ultralytics YOLOv8](https://github.com/ultralytics/yolov8):** For the powerful object detection framework.
- **[XGBoost](https://xgboost.readthedocs.io/):** For the efficient gradient boosting library.
- **[TA-Lib](https://ta-lib.org/):** For technical analysis indicators.
- **Community Contributors:** Special thanks to all contributors who have helped in enhancing this project.

---

## Troubleshooting

**1. Unable to Connect to Binance API**

- **Solution:**
  - Ensure that your API keys are correctly set in `utils/config.py`.
  - Check if your API keys have the necessary permissions.
  - Verify your internet connection.

**2. Object Detection Model Not Detecting Patterns Accurately**

- **Solution:**
  - Ensure that the annotation process was thorough and accurate.
  - Increase the size of the training dataset.
  - Experiment with different YOLOv8 model sizes (e.g., `yolov8s.pt`, `yolov8m.pt`).
  - Fine-tune hyperparameters during training.

**3. Predictive Model Low Accuracy**

- **Solution:**
  - Enhance feature engineering by incorporating more technical indicators.
  - Address class imbalance if present.
  - Experiment with different machine learning models or ensemble methods.
  - Perform hyperparameter tuning using techniques like grid search or randomized search.

**4. Trading Orders Not Executing**

- **Solution:**
  - Verify that your Binance API keys have trading permissions enabled.
  - Check if the trading symbol (`TRADE_SYMBOL`) is correct and available on Binance.
  - Ensure sufficient balance in your Binance account.
  - Review logs in `trading.log` for specific error messages.

---

## Future Enhancements

- **Real-Time Data Processing:** Implement real-time data streaming for instant pattern detection and trading decisions.
- **Enhanced Visualization:** Develop a dashboard to visualize detected patterns, trends, and trading performance.
- **Advanced Machine Learning Models:** Explore deep learning models like LSTM or Transformers for improved predictive capabilities.
- **Risk Management Features:** Incorporate advanced risk management strategies, including dynamic position sizing and portfolio diversification.
- **Backtesting Framework:** Develop a comprehensive backtesting system to evaluate trading strategies against historical data.
- **User Interface:** Create a user-friendly interface for configuring settings, monitoring performance, and managing trades.

---

**Disclaimer:** Trading cryptocurrencies involves significant risk and is not suitable for every investor. The Crypto Trend Predictor is provided for educational purposes only and should not be construed as financial advice. Use at your own risk.

---