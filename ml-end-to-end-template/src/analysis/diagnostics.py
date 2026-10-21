def diagnostics(df, target):
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": {str(k): int(v) for k, v in df[target].value_counts().items()},
        "constant_columns": [str(c) for c in df.columns if df[c].nunique(dropna=False) <= 1],
        "potential_id_columns": [str(c) for c in df.columns if c != target and df[c].nunique() == len(df)],
    }
