import pandas as pd

def validate_dataset(df: pd.DataFrame, target: str) -> dict:
    if target not in df.columns:
        raise ValueError(f"Target '{target}' nao existe. Colunas: {list(df.columns)}")
    if df[target].isna().any():
        raise ValueError("Target contem valores ausentes")
    if df[target].nunique() != 2:
        raise ValueError("Esta versao inicial suporta classificacao binaria")
    if len(df) < 20:
        raise ValueError("Dataset pequeno demais para o pipeline inicial")
    return {"rows": len(df), "columns": len(df.columns), "duplicates": int(df.duplicated().sum()),
            "missing_by_column": df.isna().sum().to_dict(), "target_counts": df[target].value_counts().to_dict()}
