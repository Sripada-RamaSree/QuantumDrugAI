"""Classical graph neural network baseline."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        h = torch.bmm(adj, x) / deg
        return self.linear(h)


class ClassicalGNN(nn.Module):
    def __init__(self, node_dim=18, desc_dim=7, hidden_dim=64, out_dim=1, dropout=0.2):
        super().__init__()
        self.conv1 = GraphConv(node_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.desc_proj = nn.Linear(desc_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def encode(self, x, adj, desc):
        h = F.relu(self.conv1(x, adj))
        h = F.relu(self.conv2(h, adj))
        h = F.relu(self.conv3(h, adj))
        graph_emb = h.mean(dim=1)
        desc_emb = F.relu(self.desc_proj(desc))
        return torch.cat([graph_emb, desc_emb], dim=-1)

    def forward(self, x, adj, desc):
        emb = self.encode(x, adj, desc)
        return self.head(emb)
