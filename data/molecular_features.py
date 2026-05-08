"""Molecular feature extraction utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, QED
    RDKIT_AVAILABLE = True
except Exception:  # pragma: no cover
    Chem = None
    Descriptors = None
    Crippen = None
    QED = None
    RDKIT_AVAILABLE = False


ATOM_TYPES = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "H", "Other"]


@dataclass
class MolecularGraph:
    smiles: str
    node_features: np.ndarray
    adjacency: np.ndarray
    descriptor: np.ndarray
    target: Optional[float] = None


def one_hot(value: str, choices: List[str]) -> List[float]:
    return [1.0 if value == c else 0.0 for c in choices]


def atom_features(atom) -> List[float]:
    symbol = atom.GetSymbol()
    if symbol not in ATOM_TYPES:
        symbol = "Other"
    return (
        one_hot(symbol, ATOM_TYPES)
        + [
            atom.GetAtomicNum() / 100.0,
            atom.GetTotalDegree() / 6.0,
            atom.GetFormalCharge() / 5.0,
            float(atom.GetHybridization()) / 10.0,
            atom.GetTotalNumHs() / 8.0,
            float(atom.GetIsAromatic()),
            atom.GetMass() / 250.0,
        ]
    )


def smiles_to_mol(smiles: str):
    if not RDKIT_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol


def molecular_descriptors(smiles: str) -> np.ndarray:
    """Return robust molecular descriptors. Uses simple string fallback without RDKit."""
    if RDKIT_AVAILABLE:
        mol = smiles_to_mol(smiles)
        if mol is not None:
            return np.array([
                Descriptors.MolWt(mol) / 500.0,
                Descriptors.TPSA(mol) / 200.0,
                Crippen.MolLogP(mol) / 10.0,
                Descriptors.NumHDonors(mol) / 10.0,
                Descriptors.NumHAcceptors(mol) / 15.0,
                Descriptors.NumRotatableBonds(mol) / 20.0,
                QED.qed(mol),
            ], dtype=np.float32)
    # Fallback: deterministic string features
    s = smiles or ""
    return np.array([
        len(s) / 100.0,
        s.count("C") / 50.0,
        s.count("N") / 20.0,
        s.count("O") / 20.0,
        s.count("=") / 20.0,
        s.count("1") / 10.0,
        sum(ord(c) for c in s) % 100 / 100.0,
    ], dtype=np.float32)


def smiles_to_graph(smiles: str, target: Optional[float] = None, max_nodes: int = 64) -> Optional[MolecularGraph]:
    """Convert SMILES to padded graph. If RDKit is unavailable, create a simple chain graph."""
    if RDKIT_AVAILABLE:
        mol = smiles_to_mol(smiles)
        if mol is None:
            return None
        atoms = list(mol.GetAtoms())[:max_nodes]
        n = len(atoms)
        x = np.zeros((max_nodes, len(ATOM_TYPES) + 7), dtype=np.float32)
        for i, atom in enumerate(atoms):
            x[i] = np.array(atom_features(atom), dtype=np.float32)
        a = np.eye(max_nodes, dtype=np.float32)
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i < max_nodes and j < max_nodes:
                a[i, j] = 1.0
                a[j, i] = 1.0
        return MolecularGraph(smiles, x, a, molecular_descriptors(smiles), target)

    # Lightweight fallback based on characters
    s = smiles[:max_nodes]
    x = np.zeros((max_nodes, len(ATOM_TYPES) + 7), dtype=np.float32)
    for i, ch in enumerate(s):
        symbol = ch if ch in ATOM_TYPES else "Other"
        x[i, : len(ATOM_TYPES)] = np.array(one_hot(symbol, ATOM_TYPES), dtype=np.float32)
        x[i, len(ATOM_TYPES)] = ord(ch) / 128.0
    a = np.eye(max_nodes, dtype=np.float32)
    for i in range(max(0, len(s) - 1)):
        a[i, i + 1] = a[i + 1, i] = 1.0
    return MolecularGraph(smiles, x, a, molecular_descriptors(smiles), target)


def valid_smiles(smiles: str) -> bool:
    if not smiles:
        return False
    if not RDKIT_AVAILABLE:
        return isinstance(smiles, str) and len(smiles) > 0
    return smiles_to_mol(smiles) is not None


def property_scores(smiles: str) -> dict:
    desc = molecular_descriptors(smiles)
    out = {
        "mol_weight_norm": float(desc[0]),
        "tpsa_norm": float(desc[1]),
        "logp_norm": float(desc[2]),
        "qed": float(desc[-1]),
    }
    if RDKIT_AVAILABLE and valid_smiles(smiles):
        mol = smiles_to_mol(smiles)
        out["logp"] = float(Crippen.MolLogP(mol))
        out["qed"] = float(QED.qed(mol))
        try:
            from rdkit.Chem import rdMolDescriptors
            out["sa_proxy"] = float(1.0 / (1.0 + Descriptors.NumRotatableBonds(mol)))
        except Exception:
            out["sa_proxy"] = 0.0
    return out
