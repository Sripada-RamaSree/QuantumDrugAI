"""Property-guided molecule optimization."""
from __future__ import annotations
import argparse
import os
import pandas as pd
from models.molecule_generator import generate_candidates, mutate_smiles
from data.molecular_features import property_scores, valid_smiles


def objective(smiles: str) -> float:
    s = property_scores(smiles)
    # Higher QED/logP balance and lower rough synthetic complexity proxy
    return float(s.get("qed", 0.0) + 0.1 * s.get("logp", s.get("logp_norm", 0.0)) + 0.2 * s.get("sa_proxy", 0.0))


def optimize(seed_smiles="CCO", steps=30, population=20, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    best = seed_smiles if valid_smiles(seed_smiles) else "CCO"
    rows = []
    for step in range(steps):
        candidates = generate_candidates(best, n=population)
        scored = [(c, objective(c), property_scores(c)) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        best, score, props = scored[0]
        row = {"step": step + 1, "smiles": best, "objective": score, **props}
        rows.append(row)
    df = pd.DataFrame(rows)
    out = os.path.join(out_dir, "optimized_molecules.csv")
    df.to_csv(out, index=False)
    print(df.tail(10).to_string(index=False))
    print(f"Saved optimization results to {out}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_smiles", default="CCO")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()
    optimize(args.seed_smiles, args.steps, args.population, args.out_dir)


if __name__ == "__main__":
    main()
