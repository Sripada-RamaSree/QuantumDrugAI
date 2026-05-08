"""Gradient-based atom importance."""
from __future__ import annotations
import torch


def atom_importance(model, batch, device="cpu"):
    model.eval()
    x = batch["x"].to(device).clone().detach().requires_grad_(True)
    adj = batch["adj"].to(device)
    desc = batch["desc"].to(device)
    out = model(x, adj, desc).sum()
    out.backward()
    scores = x.grad.abs().sum(dim=-1)
    return scores.detach().cpu().numpy()
