"""Minimal smoke tests.
Run from repository root: python -m pytest tests
"""

def test_synthetic_dataset_builds():
    from data.preprocess import build_synthetic_dataset
    df = build_synthetic_dataset("classification")
    assert {"smiles", "target"}.issubset(df.columns)
    assert len(df) > 10


def test_descriptor_shape():
    from data.molecular_features import molecular_descriptors
    assert molecular_descriptors("CCO").shape[0] == 7
