"""Plotting utilities."""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_loss(history, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.plot(history.get("train_loss", []), label="train")
    plt.plot(history.get("val_loss", []), label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_roc(y_true, logits, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    probs = 1 / (1 + np.exp(-np.asarray(logits).reshape(-1)))
    fpr, tpr, _ = roc_curve(np.asarray(y_true).reshape(-1), probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_predicted_actual(y_true, pred, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.scatter(y_true, pred, alpha=0.7)
    mn = min(min(y_true), min(pred))
    mx = max(max(y_true), max(pred))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
