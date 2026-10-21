import pandas as pd
from sklearn.inspection import permutation_importance

def permutation_importances(model, X, y, task, random_state=42):
    score = "balanced_accuracy" if task == "classification" else "neg_mean_absolute_error"
    result = permutation_importance(model, X, y, n_repeats=3, random_state=random_state, scoring=score)
    return pd.DataFrame({"feature": X.columns, "importance_mean": result.importances_mean,
                         "importance_std": result.importances_std}).sort_values("importance_mean", ascending=False)
