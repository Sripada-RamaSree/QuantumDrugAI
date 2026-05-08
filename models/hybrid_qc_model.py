"""Quantum-classical hybrid model."""
from __future__ import annotations
import torch
import torch.nn as nn
from models.classical_gnn import ClassicalGNN
from models.quantum_layer import QuantumLayer


class HybridQCModel(nn.Module):
    def __init__(self, node_dim=18, desc_dim=7, hidden_dim=64, n_qubits=4, out_dim=1, dropout=0.2):
        super().__init__()
        self.encoder = ClassicalGNN(node_dim, desc_dim, hidden_dim, out_dim=hidden_dim, dropout=dropout)
        # encoder.head returns hidden_dim representation here
        self.quantum = QuantumLayer(hidden_dim, n_qubits=n_qubits, n_layers=2)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + n_qubits, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, adj, desc):
        classical_features = self.encoder(x, adj, desc)
        q_features = self.quantum(classical_features)
        return self.fusion(torch.cat([classical_features, q_features], dim=-1))
