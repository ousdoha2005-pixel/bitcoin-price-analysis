"""
Utility functions for generating charts and plots.

This module includes helper functions to visualise various aspects of the
Bitcoin price dataset and model outputs.  Charts are saved to the specified
filename paths, which should point inside the project's `plots/` directory.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_price_trend(df: pd.DataFrame, filename: str):
    """Plot the closing price over time and save to the given filename."""
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['Close'])
    plt.title('Bitcoin Closing Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (USD)')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_target_distribution(df: pd.DataFrame, filename: str):
    """Plot the distribution of the target variable (0 vs 1)."""
    counts = df['target'].value_counts().sort_index()
    plt.figure(figsize=(4, 4))
    plt.bar(['Down (0)', 'Up (1)'], counts)
    plt.title('Distribution of Next-Day Price Movement')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_correlation_matrix(df: pd.DataFrame, filename: str):
    """Plot a correlation matrix of numeric columns."""
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=False)
    plt.title('Correlation Matrix of Numeric Features')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, filename: str, labels=None):
    """Plot a confusion matrix using seaborn heatmap."""
    if labels is None:
        labels = ['Down (0)', 'Up (1)']
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_forecast(series: pd.Series, forecast: pd.Series, filename: str):
    """
    Plot the historical series and its forecast.  The forecast should begin
    immediately after the last index of the series.
    """
    # Concatenate series and forecast with their respective index
    plt.figure(figsize=(10, 5))
    plt.plot(series.index, series, label='Historical')
    plt.plot(forecast.index, forecast, label='Forecast')
    plt.title('Bitcoin Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (USD)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float, filename: str):
    """
    Plot the Receiver Operating Characteristic (ROC) curve and save to the given filename.

    Parameters
    ----------
    fpr : array-like
        False positive rates computed by sklearn.metrics.roc_curve.
    tpr : array-like
        True positive rates computed by sklearn.metrics.roc_curve.
    auc : float
        Area under the ROC curve.
    filename : str
        Path where the plot will be saved.
    """
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})')
    # Plot the diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_feature_importance(model, feature_names: list, filename: str, max_features: int = 10):
    """
    Plot the feature importances from an XGBoost model.

    Parameters
    ----------
    model : object
        A trained model with a ``feature_importances_`` attribute (e.g. XGBClassifier).
    feature_names : list of str
        Names of the features corresponding to the importance values.
    filename : str
        Path where the bar chart will be saved.
    max_features : int, optional (default=10)
        Number of top features to display.  If the model has fewer features, all will be shown.
    """
    # Extract importances and sort them in descending order
    importances = model.feature_importances_
    # Ensure importances and feature names are arrays for indexing
    importances = np.array(importances)
    feature_names = np.array(feature_names)
    # Get indices of features sorted by importance
    sorted_idx = np.argsort(importances)[::-1]
    # Select top features
    top_idx = sorted_idx[:max_features]
    top_features = feature_names[top_idx]
    top_importances = importances[top_idx]
    # Plot bar chart
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(top_features)), top_importances[::-1])  # reverse for descending order bottom-up
    plt.yticks(range(len(top_features)), top_features[::-1])
    plt.title('Top Feature Importances (XGBoost)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()