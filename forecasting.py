"""
Long‑term forecasting utilities for Bitcoin price using exponential smoothing.

This module provides functions to resample the closing price series to daily
frequency, fit an additive Holt‑Winters exponential smoothing model and
forecast future prices.  It also computes predicted price changes relative
to the last observed value.
"""

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def load_daily_series(filepath: str) -> pd.Series:
    """Load the BTC closing price series from a CSV and resample to daily frequency."""
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')
    series = df.set_index('Date')['Close']
    daily = series.resample('D').ffill()
    return daily


def forecast_future(series: pd.Series, horizon: int = 730) -> pd.Series:
    """
    Fit an additive trend exponential smoothing model to the input series and
    forecast the specified number of steps ahead.
    """
    model = ExponentialSmoothing(series, trend='add', seasonal=None)
    fit = model.fit()
    forecast = fit.forecast(steps=horizon)
    return forecast


def predict_year_end_prices(series: pd.Series, forecast_horizon: int = 730) -> tuple:
    """
    Forecast the series and compute predicted prices at approximately one and
    two years ahead.  Returns predicted price for 1 year ahead, 2 years ahead
    and their respective shifts relative to the last observed price.
    """
    forecast = forecast_future(series, horizon=forecast_horizon)
    last_price = series.iloc[-1]
    pred_1yr = forecast.iloc[364]
    pred_2yr = forecast.iloc[729]
    shift_1yr = pred_1yr - last_price
    shift_2yr = pred_2yr - last_price
    return pred_1yr, shift_1yr, pred_2yr, shift_2yr