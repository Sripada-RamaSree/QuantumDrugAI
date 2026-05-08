"""Dataset helper.

For reproducibility and licensing clarity, this script creates the bundled synthetic CSV.
Public datasets such as QM9, HIV, BBBP, and ZINC should be downloaded from their official repositories
or MoleculeNet/Open Graph Benchmark and saved as CSV with columns: smiles,target.
"""
from __future__ import annotations
import argparse
from data.preprocess import build_synthetic_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="classification", choices=["classification", "regression"])
    parser.add_argument("--out", default="synthetic_molecules.csv")
    args = parser.parse_args()
    df = build_synthetic_dataset(args.task)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} molecules to {args.out}")


if __name__ == "__main__":
    main()
