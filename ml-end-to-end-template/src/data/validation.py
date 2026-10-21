def validate_dataframe(df, target):
    if df.empty:
        raise ValueError("Dataset vazio")
    if target not in df:
        raise ValueError(f"Target ausente: {target}")
    if df[target].isna().any():
        raise ValueError("Target possui valores ausentes")
    if df[target].nunique() < 2:
        raise ValueError("Target deve possuir pelo menos dois valores distintos")
    return {"rows": len(df), "columns": len(df.columns), "duplicates": int(df.duplicated().sum())}
