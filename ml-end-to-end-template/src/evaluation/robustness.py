import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error

def bootstrap_metric_ci(model, X, y, task, repeats=200, random_state=42):
    pred = np.asarray(model.predict(X))
    truth = np.asarray(y)
    rng = np.random.default_rng(random_state)
    metric = accuracy_score if task == "classification" else mean_absolute_error
    values = []
    for _ in range(repeats):
        ix = rng.integers(0, len(truth), len(truth))
        values.append(metric(truth[ix], pred[ix]))
    return {"lower": float(np.quantile(values, 0.025)), "upper": float(np.quantile(values, 0.975))}
