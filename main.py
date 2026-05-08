from __future__ import annotations
import argparse
from training.train_property_model import train_model


def main():
    parser = argparse.ArgumentParser(description="QuantumDrugAI molecular property prediction")
    parser.add_argument("--csv_path", default=None, help="CSV with smiles,target columns")
    parser.add_argument("--dataset", default="synthetic", help="Use synthetic unless csv_path is provided")
    parser.add_argument("--task", default="classification", choices=["classification", "regression"])
    parser.add_argument("--model", default="hybrid", choices=["classical", "hybrid"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()

    train_model(
        csv_path=args.csv_path,
        dataset_name=args.dataset,
        task=args.task,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        n_qubits=args.n_qubits,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
