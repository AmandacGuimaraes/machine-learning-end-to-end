import pandas as pd
from src.modeling.train import train_baseline
from src.modeling.evaluate import evaluate_binary

def test_training(tmp_path):
    X = pd.DataFrame({"x": list(range(40)), "group": ["a", "b"] * 20})
    y = pd.Series([0, 1] * 20)
    model = train_baseline(X, y, str(tmp_path / "model.joblib"))
    assert 0 <= evaluate_binary(model, X, y, 1)["accuracy"] <= 1
