QuantumDrugAI
Quantum–Classical Hybrid Learning Framework for Molecular Property Prediction and Molecule Optimization in Drug Discovery
QuantumDrugAI is a hybrid quantum–classical deep learning framework designed for molecular property prediction and molecule optimization in AI-driven drug discovery applications. The framework integrates Graph Neural Networks (GNNs) with Variational Quantum Circuits (VQCs) to improve molecular representation learning and predictive performance.
The project supports:
•	Molecular property prediction
•	Quantum-enhanced molecular representation learning
•	Molecule optimization and generation
•	Explainability and visualization
•	Benchmarking against classical baselines
________________________________________
Features
•	Hybrid Quantum–Classical Architecture
•	Graph Neural Network-based Molecular Encoding
•	Variational Quantum Circuit Integration
•	Molecular Graph Construction using RDKit
•	Support for QM9, HIV, BBBP, and ZINC datasets
•	Training, Evaluation, and Visualization Modules
•	Molecule Optimization Pipeline
•	Explainability Utilities
•	Fully Modular Research-Oriented Codebase
________________________________________
Project Structure
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
________________________________________
Datasets
The framework is designed to work with the following public molecular datasets.
Dataset	Task
QM9	Molecular property regression
HIV	Molecular activity classification
BBBP	Blood-brain barrier penetration prediction
ZINC	Molecule optimization and generation
________________________________________
System Requirements
Hardware
Recommended:
•	Intel i5/i7 or AMD Ryzen processor
•	16 GB RAM
•	NVIDIA GPU with CUDA support (optional but recommended)
Minimum:
•	8 GB RAM
•	CPU execution supported
________________________________________
Software Requirements
•	Python 3.10+
•	PyTorch
•	PyTorch Geometric
•	RDKit
•	PennyLane
•	Qiskit
•	NumPy
•	Pandas
•	Scikit-learn
•	Matplotlib
________________________________________
Installation
Step 1 — Clone Repository
git clone https://github.com/your-username/QuantumDrugAI.git
cd QuantumDrugAI
Step 2 — Create Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate
Linux / macOS
python3 -m venv venv
source venv/bin/activate
________________________________________
Step 3 — Install Dependencies
pip install -r requirements.txt
________________________________________
Running the Framework
Dataset Preprocessing
python data/preprocess.py
________________________________________
Train Classical GNN Model
python training/train_property_model.py --model classical
________________________________________
Train Hybrid Quantum–Classical Model
python training/train_property_model.py --model hybrid
________________________________________
Molecule Optimization
python training/optimize_molecules.py
________________________________________
Visualization and Evaluation
python evaluation/visualization.py
________________________________________
Hybrid Architecture
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
________________________________________
Quantum Computing Module
The quantum module uses variational quantum circuits to learn enhanced molecular representations.
Implemented quantum operations include:
•	Angle Embedding
•	Rotation Gates
•	Entanglement Layers
•	Pauli-Z Measurements
•	Hybrid Gradient Optimization
Quantum simulation is executed using PennyLane and Qiskit simulators.
________________________________________
Evaluation Metrics
Regression Tasks
•	MAE
•	RMSE
•	R² Score
Classification Tasks
•	Accuracy
•	Precision
•	Recall
•	F1-score
•	ROC-AUC
Molecule Optimization
•	QED
•	logP
•	Synthetic Accessibility
•	Validity
•	Novelty
•	Uniqueness
________________________________________
Explainability
The framework includes explainability utilities for molecular interpretation.
Supported features:
•	Atom Importance Analysis
•	Feature Attribution
•	Molecular Embedding Visualization
•	Attention-Based Interpretability
________________________________________
Results
Generated outputs are stored in:
results/
Outputs include:
•	Trained models
•	Evaluation reports
•	Loss curves
•	ROC curves
•	Optimized molecules
•	Embedding visualizations
________________________________________
Example Output
Epoch 20/50
Training Loss: 0.0214
Validation Accuracy: 94.82%
ROC-AUC: 0.963
________________________________________
Research Contributions
This framework contributes:
1.	Hybrid quantum–classical molecular learning
2.	Quantum-enhanced molecular representation extraction
3.	Molecular optimization for drug discovery
4.	Explainable molecular property prediction
5.	Benchmark evaluation on public datasets
________________________________________
Future Enhancements
Potential future improvements include:
•	Real quantum hardware execution
•	Transformer-based molecular encoders
•	Diffusion-based molecule generation
•	Multi-objective molecular optimization
•	Federated quantum molecular learning
________________________________________
Citation
If you use this repository in your research, please cite:
Quantum–Classical Hybrid Learning Framework for Molecular Property Prediction and Molecule Optimization in Drug Discovery
________________________________________
License
This project is intended for academic and research purposes.
________________________________________
Contact
For research collaboration or technical queries:
•	Author: Researcher
•	Project: QuantumDrugAI

