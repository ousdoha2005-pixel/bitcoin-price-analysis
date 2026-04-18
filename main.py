"""
Entry point for the Bitcoin price prediction project.

Run classification and long‑term forecasting tasks from the command line:

```
python main.py --task classify
python main.py --task forecast
```

The classification task prints ROC‑AUC scores for four models and shows a
confusion matrix for the best model.  The forecasting task reports predicted
closing prices for the end of 2026 and 2027 and their shifts relative to the
last observed price.
"""

import argparse
import os

import pandas as pd  # Needed for time delta operations in forecasting

from data_preprocessing import (
    load_data,
    clean_data,
    engineer_features,
    prepare_classification_data,
)
from classification import (
    train_models,
    evaluate_models,
    get_best_model,
    compute_confusion_matrix,
)
from forecasting import (
    load_daily_series,
    predict_year_end_prices,
    forecast_future,
)

from sklearn.metrics import roc_curve, roc_auc_score

# Import plotting utilities.  These functions generate and save charts to disk.
from plotting import (
    plot_price_trend,
    plot_target_distribution,
    plot_correlation_matrix,
    plot_confusion_matrix,
    plot_forecast,
    plot_roc_curve,
    plot_feature_importance,
)


def run_classification(data_path: str):
    """
    Perform the classification task:

    * Load and clean the dataset
    * Engineer additional features
    * Split data chronologically into training and validation sets
    * Train multiple classification models and evaluate them using ROC‑AUC
    * Identify the best model and compute a confusion matrix
    * Generate and save informative plots
    """
    # Load and preprocess the data
    df = load_data(data_path)
    df = clean_data(df)
    df = engineer_features(df)
    # Features used for classification
    feature_cols = [
        'open-close',
        'low-high',
        'is_quarter_end',
        'return',
        'volatility',
        'SMA_ratio',
        'price_SMA_diff',
    ]
    # Prepare training and validation sets
    X_train, X_valid, y_train, y_valid, scaler = prepare_classification_data(df, feature_cols)
    # Train models and evaluate
    models = train_models(X_train, y_train)
    evaluation_df = evaluate_models(models, X_train, y_train, X_valid, y_valid)
    # Display evaluation metrics
    print('\nClassification model performance (ROC‑AUC):')
    print(evaluation_df.to_string(index=False))
    # Select the best performing model
    best_name, best_model = get_best_model(models, evaluation_df)
    print(f"\nBest model: {best_name}")
    # Compute confusion matrix on validation data
    cm = compute_confusion_matrix(best_model, X_valid, y_valid)
    print('Confusion matrix on validation data (threshold=0.5):')
    print(cm)
    # Create a directory for plots if it does not already exist
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    # Generate visualisations
    # Price trend over time
    plot_price_trend(df, os.path.join(plots_dir, 'price_trend.png'))
    # Target distribution (down vs up days)
    plot_target_distribution(df, os.path.join(plots_dir, 'target_distribution.png'))
    # Correlation matrix for numeric features (include Close price and engineered features)
    numeric_df = df[feature_cols + ['Close']].copy()
    plot_correlation_matrix(numeric_df, os.path.join(plots_dir, 'correlation_matrix.png'))
    # Confusion matrix for the best model
    plot_confusion_matrix(cm, os.path.join(plots_dir, 'confusion_matrix.png'))
    # Additional analysis for XGBoost: ROC curve and feature importance
    # Check if an XGBoost model was trained and evaluate it separately
    xgb_model = models.get('XGBoost')
    if xgb_model is not None:
        # Compute ROC curve and AUC for XGBoost on the validation set
        xgb_probs = xgb_model.predict_proba(X_valid)[:, 1]
        fpr, tpr, _ = roc_curve(y_valid, xgb_probs)
        xgb_auc = roc_auc_score(y_valid, xgb_probs)
        # Plot the ROC curve
        plot_roc_curve(
            fpr,
            tpr,
            xgb_auc,
            os.path.join(plots_dir, 'xgboost_roc_curve.png'),
        )
        # Plot feature importances for XGBoost
        plot_feature_importance(
            xgb_model,
            feature_cols,
            os.path.join(plots_dir, 'xgboost_feature_importance.png'),
        )


def run_forecast(data_path: str):
    """
    Perform the long‑term forecasting task:

    * Load the closing price series and resample to daily frequency
    * Fit an exponential smoothing model and forecast two years ahead
    * Report predicted prices at one and two years ahead and their shifts
    * Generate and save a plot of the historical series with the forecast
    """
    # Load the series and resample to daily frequency
    series = load_daily_series(data_path)
    # Generate a forecast for the specified horizon (two years = 730 days)
    horizon = 730
    forecast = forecast_future(series, horizon=horizon)
    # Compute year‑ahead predictions and their shifts
    pred_1yr, shift_1yr, pred_2yr, shift_2yr = predict_year_end_prices(series, forecast_horizon=horizon)
    # Get last observed date and price
    last_date = series.index[-1]
    last_price = series.iloc[-1]
    # Print summary
    print(f"Last observed date: {last_date.date()} with closing price ${last_price:,.2f}")
    print(
        f"Predicted closing price ~1 year ahead ({(last_date + pd.Timedelta(days=365)).strftime('%Y-%m-%d')}): $"
        f"{pred_1yr:,.2f} (change: ${shift_1yr:,.2f})"
    )
    print(
        f"Predicted closing price ~2 years ahead ({(last_date + pd.Timedelta(days=730)).strftime('%Y-%m-%d')}): $"
        f"{pred_2yr:,.2f} (change: ${shift_2yr:,.2f})"
    )
    # Create plots directory if necessary and plot the forecast alongside the historical series
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    plot_forecast(series, forecast, os.path.join(plots_dir, 'forecast.png'))


def main():
    parser = argparse.ArgumentParser(description='Bitcoin price prediction project')
    # If no task is specified on the command line, default to 'classify'
    parser.add_argument(
        '--task',
        choices=['classify', 'forecast'],
        default='classify',
        help="Task to run: 'classify' trains and evaluates classification models, 'forecast' performs long‑term forecasting. Defaults to 'classify' if omitted."
    )
    parser.add_argument('--data', default=os.path.join('data', 'BTC-USD_clean.csv'), help='Path to the cleaned dataset')
    args = parser.parse_args()
    if args.task == 'classify':
        run_classification(args.data)
    elif args.task == 'forecast':
        run_forecast(args.data)


if __name__ == '__main__':
    main()