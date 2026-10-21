import pandas as pd
import pytest
from sklearn.datasets import make_classification
from src.data.validation import validate_dataframe
from src.features.splitting import split_data
from src.modeling.train import make_pipeline
from src.evaluation.metrics import calculate_metrics
from src.monitoring.drift import numeric_drift

def sample():
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    df = pd.DataFrame(X, columns=list("abcd"))
    df["target"] = y
    return df

def test_validation():
    assert validate_dataframe(sample(), "target")["rows"] == 100

def test_empty():
    with pytest.raises(ValueError):
        validate_dataframe(pd.DataFrame(), "target")

def test_training():
    cfg = {"data": {"target": "target"}, "validation": {"strategy": "stratified", "test_size": .2, "random_state": 42}}
    Xtr, Xte, ytr, yte = split_data(sample(), cfg)
    model = make_pipeline(Xtr, "classification").fit(Xtr, ytr)
    assert 0 <= calculate_metrics(model, Xte, yte, "classification")["accuracy"] <= 1

def test_drift():
    df = sample()
    assert len(numeric_drift(df, df)) == 5
