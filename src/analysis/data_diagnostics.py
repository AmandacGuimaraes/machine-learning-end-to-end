import pandas as pd

def diagnose(df: pd.DataFrame, target: str) -> dict:
    if target not in df.columns:
        raise ValueError(f"Target ausente: {target}")
    if df.empty:
        raise ValueError("Dataset vazio")
    counts = df[target].value_counts(dropna=False)
    return {
        "rows": int(len(df)), "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_missing": int(df[target].isna().sum()),
        "target_distribution": {str(k): int(v) for k, v in counts.items()},
        "constant_columns": [str(c) for c in df.columns if df[c].nunique(dropna=False) <= 1],
        "potential_id_columns": [str(c) for c in df.columns if c != target and df[c].nunique(dropna=False) == len(df)],
    }
