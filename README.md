# QuantumDrugAI

A complete reference implementation for **Quantum–Classical Hybrid Learning Framework for Molecular Property Prediction and Molecule Optimization in Drug Discovery**.

The project supports:

- Molecular property prediction for regression and classification
- Classical graph neural network baseline
- Quantum–classical hybrid model with optional PennyLane layer
- RDKit-based molecular feature extraction
- Synthetic fallback dataset for quick testing
- Molecule optimization using property-guided mutation
- Evaluation metrics, plots, and atom-level attribution

## Installation

```bash
pip install -r requirements.txt
```

RDKit installation may be easier with Conda:

```bash
conda install -c conda-forge rdkit
pip install torch pennylane scikit-learn pandas matplotlib tqdm
```

## Quick Test Without Downloading Datasets

```bash
python main.py --dataset synthetic --task classification --epochs 5
```

Regression example:

```bash
python main.py --dataset synthetic --task regression --epochs 5
```

## Use CSV Dataset

Your CSV should contain at least:

```text
smiles,target
CCO,0.4
CCN,0.7
```

Run:

```bash
python main.py --csv_path path/to/data.csv --task regression --epochs 30
```

For classification:

```bash
python main.py --csv_path path/to/data.csv --task classification --epochs 30
```

## Molecule Optimization

```bash
python -m training.optimize_molecules --seed_smiles CCO --steps 20
```

## Outputs

Results are saved in `results/`:

- trained model weights
- metrics JSON
- loss curves
- ROC curves for classification
- predicted-vs-actual plots for regression
- optimized molecule table

## Project Structure

```text
QuantumDrugAI/
├── data/
│   ├── download_datasets.py
│   ├── molecular_features.py
│   └── preprocess.py
├── models/
│   ├── classical_gnn.py
│   ├── quantum_layer.py
│   ├── hybrid_qc_model.py
│   └── molecule_generator.py
├── training/
│   ├── train_property_model.py
│   ├── train_generator.py
│   └── optimize_molecules.py
├── evaluation/
│   ├── metrics.py
│   ├── compare_models.py
│   └── visualization.py
├── explainability/
│   ├── atom_importance.py
│   └── feature_attribution.py
├── main.py
└── README.md
```

## Notes

The quantum layer automatically falls back to a differentiable classical surrogate when PennyLane is unavailable, so the project remains executable in standard Python environments.
