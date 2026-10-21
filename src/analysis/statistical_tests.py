import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal

def categorical_association(df: pd.DataFrame, feature: str, target: str) -> dict:
    tab = pd.crosstab(df[feature], df[target])
    if min(tab.shape) < 2:
        raise ValueError("Variaveis sem categorias suficientes")
    chi2, p, dof, expected = chi2_contingency(tab)
    n = tab.to_numpy().sum()
    v = np.sqrt(chi2 / (n * min(tab.shape[0] - 1, tab.shape[1] - 1)))
    return {"test": "chi_square", "statistic": float(chi2), "p_value": float(p),
            "dof": int(dof), "min_expected": float(expected.min()),
            "cramers_v": float(v), "warning": "Frequencias esperadas baixas; interpretar com cautela" if (expected < 5).mean() > .2 else ""}

def numeric_association(df: pd.DataFrame, feature: str, target: str) -> dict:
    subset = df[[feature, target]].dropna()
    groups = [part[feature].to_numpy(dtype=float) for _, part in subset.groupby(target)]
    if len(groups) < 2 or any(len(g) < 2 for g in groups):
        raise ValueError("Grupos insuficientes")
    if len(groups) == 2:
        stat, p = mannwhitneyu(groups[0], groups[1], alternative="two-sided")
        return {"test": "mann_whitney_u", "statistic": float(stat), "p_value": float(p),
                "effect_size_rank_biserial_abs": float(abs(1 - 2 * stat / (len(groups[0]) * len(groups[1]))))}
    stat, p = kruskal(*groups)
    return {"test": "kruskal_wallis", "statistic": float(stat), "p_value": float(p)}

def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    return adjusted.tolist()
