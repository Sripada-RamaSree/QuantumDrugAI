"""Training and evaluation loop."""
from __future__ import annotations
import json
import os
from typing import Dict, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

from data.preprocess import MolecularDataset, load_dataset, make_splits
from models.classical_gnn import ClassicalGNN
from models.hybrid_qc_model import HybridQCModel
from evaluation.metrics import classification_metrics, regression_metrics
from evaluation.visualization import plot_loss, plot_roc, plot_predicted_actual


def batch_to_device(batch, device):
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


def build_model(model_name: str, node_dim: int, desc_dim: int, task: str, hidden_dim=64, n_qubits=4):
    out_dim = 1
    if model_name == "classical":
        return ClassicalGNN(node_dim=node_dim, desc_dim=desc_dim, hidden_dim=hidden_dim, out_dim=out_dim)
    if model_name == "hybrid":
        return HybridQCModel(node_dim=node_dim, desc_dim=desc_dim, hidden_dim=hidden_dim, n_qubits=n_qubits, out_dim=out_dim)
    raise ValueError("model_name must be classical or hybrid")


def run_epoch(model, loader, optimizer, criterion, task, device, train=True):
    model.train(train)
    total_loss = 0.0
    y_all, p_all = [], []
    for batch in loader:
        batch = batch_to_device(batch, device)
        if train:
            optimizer.zero_grad()
        logits = model(batch["x"], batch["adj"], batch["desc"])
        y = batch["y"]
        loss = criterion(logits, y)
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total_loss += float(loss.item()) * y.size(0)
        y_all.extend(y.detach().cpu().numpy().reshape(-1).tolist())
        p_all.extend(logits.detach().cpu().numpy().reshape(-1).tolist())
    return total_loss / len(loader.dataset), np.array(y_all), np.array(p_all)


def train_model(
    csv_path=None,
    dataset_name="synthetic",
    task="classification",
    model_name="hybrid",
    epochs=20,
    batch_size=32,
    lr=1e-3,
    hidden_dim=64,
    n_qubits=4,
    out_dir="results",
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    df = load_dataset(csv_path=csv_path, dataset=dataset_name, task=task)
    ds = MolecularDataset(df, task=task)
    train_ds, val_ds, test_ds = make_splits(ds, seed=seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    sample = ds[0]
    node_dim = sample["x"].shape[-1]
    desc_dim = sample["desc"].shape[-1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(model_name, node_dim, desc_dim, task, hidden_dim, n_qubits).to(device)
    criterion = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_path = os.path.join(out_dir, f"{model_name}_{task}_best.pt")

    for _ in tqdm(range(epochs), desc="Training"):
        tr_loss, _, _ = run_epoch(model, train_loader, optimizer, criterion, task, device, train=True)
        va_loss, _, _ = run_epoch(model, val_loader, optimizer, criterion, task, device, train=False)
        scheduler.step(va_loss)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, y_true, preds = run_epoch(model, test_loader, optimizer, criterion, task, device, train=False)
    metrics = classification_metrics(y_true, preds) if task == "classification" else regression_metrics(y_true, preds)
    metrics["test_loss"] = float(test_loss)
    metrics["n_samples"] = len(ds)
    metrics["model"] = model_name
    metrics["task"] = task

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    plot_loss(history, os.path.join(out_dir, "loss_curve.png"))
    if task == "classification" and len(set(y_true.tolist())) > 1:
        plot_roc(y_true, preds, os.path.join(out_dir, "roc_curve.png"))
    if task == "regression":
        plot_predicted_actual(y_true.tolist(), preds.tolist(), os.path.join(out_dir, "predicted_vs_actual.png"))
    print(json.dumps(metrics, indent=2))
    return model, metrics
