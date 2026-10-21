from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def calculate_metrics(model, X, y, task):
    pred = model.predict(X)
    if task == "classification":
        return {"accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0))}
    return {"mae": float(mean_absolute_error(y, pred)), "rmse": float(np.sqrt(mean_squared_error(y, pred))), "r2": float(r2_score(y, pred))}
