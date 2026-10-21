from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.features.engineering import make_preprocessor

def train_baseline(X_train, y_train, output_path: str):
    model = Pipeline([
        ("preprocess", make_preprocessor(X_train)),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    model.fit(X_train, y_train)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return model
