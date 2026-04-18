"""
Data preprocessing and feature engineering utilities for the Bitcoin price prediction project.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(filepath: str) -> pd.DataFrame:
    """Read the CSV file containing Bitcoin price data."""
    df = pd.read_csv(filepath)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the Date column to datetime, drop missing dates, sort and remove duplicates."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.drop_duplicates(subset='Date')
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional features from the raw data to aid classification.

    Adds the following columns:
        - open-close: difference between open and close price
        - low-high: difference between low and high price
        - is_quarter_end: flag for last month of each quarter
        - return: daily percentage change of closing price
        - volatility: 7‑day rolling standard deviation of return
        - SMA_7, SMA_21: 7‑ and 21‑day simple moving averages
        - SMA_ratio: ratio of SMA_7 to SMA_21
        - price_SMA_diff: difference between close and SMA_7
        - target: label indicating whether next day's close is higher than today's
    """
    df = df.copy()
    # Base differences
    df['open-close'] = df['Open'] - df['Close']
    df['low-high'] = df['Low'] - df['High']
    # Quarter end indicator
    df['month'] = df['Date'].dt.month
    df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)
    # Returns and volatility
    df['return'] = df['Close'].pct_change()
    df['volatility'] = df['return'].rolling(window=7).std()
    # Moving averages
    df['SMA_7'] = df['Close'].rolling(window=7).mean()
    df['SMA_21'] = df['Close'].rolling(window=21).mean()
    df['SMA_ratio'] = df['SMA_7'] / df['SMA_21']
    df['price_SMA_diff'] = df['Close'] - df['SMA_7']
    # Target variable: 1 if next day's close > today's close, else 0
    df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    return df


def prepare_classification_data(df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    Prepare data for classification.

    Drops rows with NaNs in feature columns or target, standardises features using
    StandardScaler and returns a time‑based train/validation split (70%/30%).

    Returns:
        X_train, X_valid, y_train, y_valid, scaler
    """
    df_model = df.dropna(subset=feature_cols + ['target']).reset_index(drop=True)
    X = df_model[feature_cols]
    y = df_model['target'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    split_index = int(0.7 * len(df_model))
    X_train = X_scaled[:split_index]
    X_valid = X_scaled[split_index:]
    y_train = y[:split_index]
    y_valid = y[split_index:]
    return X_train, X_valid, y_train, y_valid, scaler