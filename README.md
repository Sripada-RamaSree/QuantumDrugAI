[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20082788.svg)](https://doi.org/10.5281/zenodo.20082788)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)

---

## Code Archive

GitHub repository: https://github.com/Sripada-RamaSree/QuantumDrugAI

Permanent archived release (Zenodo DOI): https://doi.org/10.5281/zenodo.20082788
# QuantumDrugAI

### Quantum–Classical Hybrid Learning Framework for Molecular Property Prediction and Molecule Optimization in Drug Discovery

## Overview

QuantumDrugAI is a hybrid quantum–classical deep learning framework designed for molecular property prediction and molecule optimization in AI-driven drug discovery applications. The framework integrates Graph Neural Networks (GNNs) with Variational Quantum Circuits (VQCs) to improve molecular representation learning, predictive performance, and optimization efficiency.

The system supports molecular graph construction, quantum-enhanced feature learning, explainability analysis, and scalable pharmaceutical AI workflows for computational chemistry and drug discovery research.

---

## Key Features

* Hybrid Quantum–Classical Architecture
* Graph Neural Network-based Molecular Encoding
* Variational Quantum Circuit Integration
* Molecular Graph Construction using RDKit
* Support for QM9, HIV, BBBP, and ZINC datasets
* Training, Evaluation, and Visualization Modules
* Molecule Optimization Pipeline
* Explainability Utilities
* Fully Modular Research-Oriented Codebase

---

## System Architecture

The framework follows a hybrid molecular learning pipeline:

```text id="if17y5"
SMILES Input
      ↓
Molecular Graph Construction
      ↓
Graph Neural Network Encoder
      ↓
Dense Projection Layer
      ↓
Variational Quantum Circuit
      ↓
Feature Fusion Layer
      ↓
Property Prediction / Optimization Output
```

---

## Repository Structure

```text id="mbgw70"
QuantumDrugAI/
│
├── data/
│   ├── download_datasets.py
│   ├── preprocess.py
│   └── molecular_features.py
│
├── models/
│   ├── classical_gnn.py
│   ├── quantum_layer.py
│   ├── hybrid_qc_model.py
│   └── molecule_generator.py
│
├── training/
│   ├── train_property_model.py
│   ├── train_generator.py
│   └── optimize_molecules.py
│
├── evaluation/
│   ├── metrics.py
│   ├── compare_models.py
│   └── visualization.py
│
├── explainability/
│   ├── atom_importance.py
│   └── feature_attribution.py
│
├── results/
├── requirements.txt
├── README.md
└── main.py
```

---

## Installation

Clone the repository:

```text id="6d5azk"
git clone https://github.com/Sripada-RamaSree/QuantumDrugAI
cd QuantumDrugAI
```

Create environment:

```text id="mq9g1m"
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```text id="5m6w9r"
pip install -r requirements.txt
```

---

## Dataset Preparation

Supported datasets:

* QM9 → Molecular property regression
* HIV → Molecular activity classification
* BBBP → Blood-brain barrier penetration prediction
* ZINC → Molecule optimization and generation

Run preprocessing:

```text id="gl36ri"
python data/preprocess.py
```

---

## Training

Train Classical GNN Model:

```text id="h8r7sk"
python training/train_property_model.py --model classical
```

Train Hybrid Quantum–Classical Model:

```text id="3kpnm4"
python training/train_property_model.py --model hybrid
```

Molecule Optimization:

```text id="uvm2gf"
python training/optimize_molecules.py
```

---

## Evaluation

Run evaluation and visualization:

```text id="l3hnsw"
python evaluation/visualization.py
```

Metrics:

### Regression Tasks

* MAE
* RMSE
* R² Score

### Classification Tasks

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

### Molecule Optimization

* QED
* logP
* Validity
* Novelty
* Uniqueness

---

## Explainability

Supported explainability features:

* Atom Importance Analysis
* Feature Attribution
* Molecular Embedding Visualization
* Attention-Based Interpretability

---

## Results Output

Generated outputs are stored in:

```text id="pt6k5k"
results/
```

Outputs include:

* Trained models
* Evaluation reports
* Loss curves
* ROC curves
* Optimized molecules
* Embedding visualizations

---

## Example Output

```text id="v17s5x"
Epoch 20/50
Training Loss: 0.0214
Validation Accuracy: 94.82%
ROC-AUC: 0.963
```

---

## Reproducibility

To reproduce experiments:

```text id="e8v1mk"
python main.py
```

---

## Future Enhancements

* Real quantum hardware execution
* Transformer-based molecular encoders
* Diffusion-based molecule generation
* Multi-objective molecular optimization
* Federated quantum molecular learning

---

## Citation

```text id="f0m2mj"
@article{QuantumDrugAI2026,
title={QuantumDrugAI: Quantum–Classical Hybrid Learning Framework for Molecular Property Prediction and Molecule Optimization in Drug Discovery},
author={Author Name},
journal={Journal Name},
year={2026}
}
```

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
