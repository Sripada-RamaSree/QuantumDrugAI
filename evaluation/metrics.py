"""Evaluation metrics."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score


def classification_metrics(y_true, logits):
    y_true = np.asarray(y_true).reshape(-1)
    probs = 1 / (1 + np.exp(-np.asarray(logits).reshape(-1)))
    preds = (probs >= 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, probs))
    except Exception:
        out["roc_auc"] = float("nan")
    return out


def regression_metrics(y_true, pred):
    y_true = np.asarray(y_true).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(mean_squared_error(y_true, pred) ** 0.5),
        "r2": float(r2_score(y_true, pred)),
    }
