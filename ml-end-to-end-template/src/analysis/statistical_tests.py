import pandas as pd
from scipy.stats import spearmanr, chi2_contingency, mannwhitneyu, kruskal

def associations(df, target):
    results = []
    y = df[target]
    for col in df.columns.drop(target):
        x = df[col]
        valid = x.notna() & y.notna()
        x, yy = x[valid], y[valid]
        if len(x) < 5 or x.nunique() < 2 or yy.nunique() < 2:
            continue
        try:
            if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(yy):
                stat, p = spearmanr(x, yy)
                test = "spearman"
            elif pd.api.types.is_numeric_dtype(x) and yy.nunique() <= 10:
                groups = [x[yy == c].to_numpy() for c in yy.unique()]
                if any(len(g) < 2 for g in groups):
                    continue
                stat, p = mannwhitneyu(*groups) if len(groups) == 2 else kruskal(*groups)
                test = "mannwhitney" if len(groups) == 2 else "kruskal"
            else:
                table = pd.crosstab(x, yy)
                if table.shape[0] < 2 or table.shape[1] < 2:
                    continue
                stat, p, _, _ = chi2_contingency(table)
                test = "chi_square"
            results.append({"feature": col, "test": test, "statistic": stat, "p_value": p})
        except (ValueError, TypeError):
            continue
    result = pd.DataFrame(results, columns=["feature", "test", "statistic", "p_value"])
    if len(result):
        # Bonferroni: conservador, evita interpretações ingênuas de múltiplos testes
        result["p_adjusted"] = (result["p_value"] * len(result)).clip(upper=1)
    return result
