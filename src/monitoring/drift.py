import pandas as pd

def missing_rate_shift(reference: pd.DataFrame, current: pd.DataFrame) -> pd.Series:
    common = reference.columns.intersection(current.columns)
    return current[common].isna().mean() - reference[common].isna().mean()
