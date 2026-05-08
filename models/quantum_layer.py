"""Quantum variational layer with safe fallback."""
from __future__ import annotations
import torch
import torch.nn as nn

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except Exception:  # pragma: no cover
    qml = None
    PENNYLANE_AVAILABLE = False


class QuantumLayer(nn.Module):
    """Differentiable quantum layer.

    If PennyLane is unavailable, uses a small trainable nonlinear layer as a surrogate, preserving
    the model interface and allowing the complete project to run anywhere.
    """
    def __init__(self, input_dim: int, n_qubits: int = 4, n_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.pre = nn.Linear(input_dim, n_qubits)

        if PENNYLANE_AVAILABLE:
            dev = qml.device("default.qubit", wires=n_qubits)

            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def circuit(inputs, weights):
                qml.AngleEmbedding(inputs, wires=range(n_qubits))
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

            weight_shapes = {"weights": (n_layers, n_qubits, 3)}
            self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
            self.surrogate = None
        else:
            self.qlayer = None
            self.surrogate = nn.Sequential(
                nn.Linear(n_qubits, n_qubits * 2),
                nn.Tanh(),
                nn.Linear(n_qubits * 2, n_qubits),
                nn.Tanh(),
            )

    def forward(self, features):
        z = torch.tanh(self.pre(features))
        if self.qlayer is not None:
            return self.qlayer(z)
        return self.surrogate(z)
