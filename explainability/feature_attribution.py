"""Descriptor-level feature attribution."""
from __future__ import annotations
import torch


def descriptor_attribution(model, batch, device="cpu"):
    model.eval()
    x = batch["x"].to(device)
    adj = batch["adj"].to(device)
    desc = batch["desc"].to(device).clone().detach().requires_grad_(True)
    out = model(x, adj, desc).sum()
    out.backward()
    return desc.grad.detach().abs().cpu().numpy()
