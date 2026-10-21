import pandas as pd
from scipy.stats import ks_2samp

def numeric_drift(reference, current, alpha=0.05):
    results = []
    for col in reference.select_dtypes(include="number").columns.intersection(current.columns):
        a = reference[col].dropna()
        b = current[col].dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        statistic, p = ks_2samp(a, b)
        results.append({"feature": col, "ks_statistic": statistic, "p_value": p, "drift_flag": bool(p < alpha)})
    return pd.DataFrame(results)
