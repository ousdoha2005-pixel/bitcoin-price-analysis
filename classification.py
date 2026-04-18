"""
Classification models for predicting short‑term Bitcoin price direction.

This module defines functions to train several classifiers (Logistic Regression,
Support Vector Classifier, RandomForest and XGBoost) on engineered features and
evaluate them using ROC‑AUC.  The dataset should be split chronologically
before calling these functions to avoid information leakage.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix


def train_models(X_train, y_train) -> dict:
    """Train a collection of classifiers and return them as a dictionary."""
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'SVC_poly': SVC(kernel='poly', probability=True),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(enable_categorical=False)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models


def evaluate_models(models: dict, X_train, y_train, X_valid, y_valid) -> pd.DataFrame:
    """
    Compute ROC‑AUC scores for each model on training and validation sets.
    Returns a DataFrame summarising the results.
    """
    results = []
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            train_scores = model.predict_proba(X_train)[:, 1]
            valid_scores = model.predict_proba(X_valid)[:, 1]
        else:
            train_decision = model.decision_function(X_train)
            valid_decision = model.decision_function(X_valid)
            # Normalise decision values to [0,1]
            train_scores = (train_decision - train_decision.min()) / (train_decision.max() - train_decision.min())
            valid_scores = (valid_decision - valid_decision.min()) / (valid_decision.max() - valid_decision.min())
        train_auc = roc_auc_score(y_train, train_scores)
        valid_auc = roc_auc_score(y_valid, valid_scores)
        results.append({'Model': name, 'Train ROC‑AUC': round(train_auc, 3), 'Validation ROC‑AUC': round(valid_auc, 3)})
    return pd.DataFrame(results)


def get_best_model(models: dict, evaluation_df: pd.DataFrame):
    """
    Identify the model with the highest validation ROC‑AUC and return its name and instance.
    """
    best_name = evaluation_df.sort_values('Validation ROC‑AUC', ascending=False)['Model'].iloc[0]
    best_model = models[best_name]
    return best_name, best_model


def compute_confusion_matrix(model, X_valid, y_valid, threshold: float = 0.5):
    """
    Compute confusion matrix for a trained model on the validation set at a given threshold.
    """
    # Get predicted probabilities or decision scores
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_valid)[:, 1]
    else:
        scores = model.decision_function(X_valid)
        probs = (scores - scores.min()) / (scores.max() - scores.min())
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_valid, preds)
    return cm