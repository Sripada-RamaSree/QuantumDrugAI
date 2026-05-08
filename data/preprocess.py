"""Dataset loading and preprocessing."""
from __future__ import annotations

from typing import Optional, Tuple
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split

from data.molecular_features import smiles_to_graph, valid_smiles, molecular_descriptors


SYNTHETIC_SMILES = [
    "CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O", "CCOC", "CCCl", "CCBr", "CC(C)O", "CC(C)N",
    "COC", "CN(C)C", "CCS", "CCP", "O=C=O", "CC(C)(C)O", "C1CCCCC1", "CCN(CC)CC", "CCOC(=O)C", "CC(C)C(=O)O",
] * 25


class MolecularDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, task: str = "classification", max_nodes: int = 64):
        self.task = task
        self.max_nodes = max_nodes
        self.graphs = []
        for _, row in frame.iterrows():
            smi = str(row["smiles"])
            if not valid_smiles(smi):
                continue
            target = float(row["target"])
            g = smiles_to_graph(smi, target=target, max_nodes=max_nodes)
            if g is not None:
                self.graphs.append(g)
        if len(self.graphs) == 0:
            raise ValueError("No valid molecules found. Check SMILES column and RDKit installation.")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        y_dtype = torch.float32
        return {
            "smiles": g.smiles,
            "x": torch.tensor(g.node_features, dtype=torch.float32),
            "adj": torch.tensor(g.adjacency, dtype=torch.float32),
            "desc": torch.tensor(g.descriptor, dtype=torch.float32),
            "y": torch.tensor([g.target], dtype=y_dtype),
        }


def build_synthetic_dataset(task: str = "classification") -> pd.DataFrame:
    rows = []
    for smi in SYNTHETIC_SMILES:
        d = molecular_descriptors(smi)
        raw = 0.35 * d[0] + 0.25 * d[2] + 0.40 * d[-1]
        if task == "classification":
            target = 1.0 if raw > 0.28 else 0.0
        else:
            target = float(raw)
        rows.append({"smiles": smi, "target": target})
    return pd.DataFrame(rows)


def load_dataset(csv_path: Optional[str] = None, dataset: str = "synthetic", task: str = "classification") -> pd.DataFrame:
    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        if dataset != "synthetic":
            raise ValueError("Automatic dataset download is not bundled. Use --csv_path or --dataset synthetic.")
        df = build_synthetic_dataset(task)
    required = {"smiles", "target"}
    if not required.issubset(df.columns):
        raise ValueError("Dataset must contain columns: smiles,target")
    return df[["smiles", "target"]].dropna().reset_index(drop=True)


def make_splits(dataset: MolecularDataset, train_ratio=0.7, val_ratio=0.15, seed=42):
    n = len(dataset)
    train_n = int(n * train_ratio)
    val_n = int(n * val_ratio)
    test_n = n - train_n - val_n
    gen = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_n, val_n, test_n], generator=gen)
