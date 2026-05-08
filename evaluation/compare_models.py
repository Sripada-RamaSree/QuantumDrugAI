"""Model comparison helper."""
from __future__ import annotations
import json
from pathlib import Path


def save_comparison(metrics_by_model: dict, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics_by_model, f, indent=2)
