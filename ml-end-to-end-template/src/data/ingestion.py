from pathlib import Path
import pandas as pd

def read_data(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("Use CSV ou Parquet (instale pyarrow para Parquet)")
