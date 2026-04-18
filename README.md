# Bitcoin Price Prediction Project

This project provides a complete, ready‑to‑run pipeline for analysing and forecasting Bitcoin prices using historical data.  It includes data preprocessing, exploratory feature engineering, classification models to predict short‑term price direction and a long‑term forecasting module to estimate year‑ahead prices.

## Directory structure

```
bitcoin_price_project/
├── data/
│   └── BTC-USD_clean.csv        # Cleaned daily Bitcoin price dataset (2014‑09‑17 to 2025‑12‑20)
├── data_preprocessing.py        # Functions for loading, cleaning and feature engineering
├── classification.py            # Classification models and evaluation
├── forecasting.py               # Long‑term forecasting using exponential smoothing
├── main.py                      # Entry point demonstrating classification and forecasting
├── plotting.py                  # Helper functions to generate charts and save them to `plots/`
├── requirements.txt             # Python dependencies
└── README.md                    # Project description and instructions
```
## Getting started

After running any task, charts are saved to a `plots/` directory created automatically in the project root.

## Getting started

1. **Install dependencies:** Create a virtual environment (optional) and install packages:

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run the classification example:**

   ```bash
   python main.py --task classify
   ```

   This will load the dataset, engineer features, train four classifiers (logistic regression, SVC, random forest and XGBoost) using a chronological split, display ROC‑AUC scores and show a confusion matrix for the best model.

   When the classification task completes, several charts are automatically generated and saved to the `plots/` directory:

   - **price_trend.png** – closing price over time
   - **target_distribution.png** – counts of up vs. down days in the target variable
   - **correlation_matrix.png** – correlation heatmap of numeric features
   - **confusion_matrix.png** – confusion matrix of the best classifier
   - **xgboost_roc_curve.png** – ROC curve for the XGBoost model along with its area under the curve
   - **xgboost_feature_importance.png** – bar chart of the most important features according to XGBoost

3. **Run the forecasting example:**

   ```bash
   python main.py --task forecast
   ```

   This will fit an exponential smoothing model to the closing price series and forecast it two years into the future, reporting predicted prices for approximately the end of 2026 and 2027.

   It also generates a plot (`forecast.png` in the `plots/` directory) overlaying the historical closing price and the forecasted values.

## Data

The `data/BTC-USD_clean.csv` file contains daily Bitcoin price data from **17 September 2014** to **20 December 2025**.  It was created by merging publicly available historical records from multiple sources (Yahoo Finance, Binance, Bitget)【382005442148099†L109-L133】, converting the `Date` column to a datetime index, sorting chronologically and removing duplicates.  The columns include:

- `Date` – date of the observation
- `Open`, `High`, `Low`, `Close` – daily OHLC prices (USD)
- `Volume` – traded volume

## Feature engineering

The project derives several features from the raw data to capture price dynamics:

- **open–close**: difference between open and close price on the same day  
- **low–high**: difference between low and high price  
- **is_quarter_end**: indicates the last month of each quarter  
- **return**: daily percentage change of the closing price  
- **volatility**: 7‑day rolling standard deviation of returns  
- **SMA_ratio**: ratio of the 7‑day to 21‑day simple moving averages  
- **price_SMA_diff**: difference between the closing price and its 7‑day SMA

These features are used by the classification models.

## Classification

The module trains four classifiers:

| Model | Description |
|------|-------------|
| **Logistic Regression** | Linear classifier with probabilistic output |
| **SVC (Polynomial)** | Support‑vector classifier with a polynomial kernel |
| **Random Forest** | Ensemble of decision trees, capturing non‑linear patterns |
| **XGBoost** | Gradient‑boosted decision trees (requires `xgboost` package) |

The dataset is split chronologically (70 % train / 30 % validation) to prevent leakage.  ROC‑AUC scores are reported for training and validation, and the confusion matrix of the best model is displayed.

## Forecasting

The `forecasting.py` module fits an additive Holt‑Winters exponential smoothing model to the daily closing price and forecasts future values.  The main function in `main.py` demonstrates how to forecast the closing price approximately one and two years ahead.  Predicted price shifts relative to the last observed value are reported.

## License

This project is provided for educational purposes.  Use at your own risk; cryptocurrency markets are volatile and subject to unpredictable influences.