from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from src.features.engineering import build_preprocessor

def make_pipeline(X, task, algorithm="random_forest", random_state=42):
    models = {
        "classification": {
            "random_forest": RandomForestClassifier(n_estimators=150, random_state=random_state, n_jobs=-1),
            "linear": LogisticRegression(max_iter=2000),
        },
        "regression": {
            "random_forest": RandomForestRegressor(n_estimators=150, random_state=random_state, n_jobs=-1),
            "linear": Ridge(),
        },
    }
    return Pipeline([("preprocess", build_preprocessor(X)), ("model", models[task][algorithm])])
