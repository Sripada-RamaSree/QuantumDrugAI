"""Simple molecule mutation and candidate generation."""
from __future__ import annotations
import random
from typing import List
from data.molecular_features import valid_smiles

TOKENS = ["C", "N", "O", "S", "F", "Cl", "Br"]


def mutate_smiles(smiles: str) -> str:
    if not smiles:
        return "C"
    ops = ["append", "replace", "delete"]
    op = random.choice(ops)
    s = smiles
    if op == "append":
        s = s + random.choice(TOKENS)
    elif op == "replace" and len(s) > 0:
        i = random.randrange(len(s))
        s = s[:i] + random.choice(TOKENS) + s[i + 1:]
    elif op == "delete" and len(s) > 1:
        i = random.randrange(len(s))
        s = s[:i] + s[i + 1:]
    return s if valid_smiles(s) else smiles


def generate_candidates(seed_smiles: str, n: int = 20) -> List[str]:
    candidates = {seed_smiles}
    current = seed_smiles
    for _ in range(n * 5):
        current = mutate_smiles(current)
        if valid_smiles(current):
            candidates.add(current)
        if len(candidates) >= n:
            break
    return list(candidates)
