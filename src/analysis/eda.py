from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_eda_report(df: pd.DataFrame, target: str, output_dir: str = "reports") -> None:
    if target not in df.columns:
        raise ValueError(f"Target ausente: {target}")
    out = Path(output_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    df.describe(include="all").T.to_csv(out / "metrics" / "describe.csv")
    missing = pd.DataFrame({"count": df.isna().sum(), "percent": df.isna().mean().mul(100)})
    missing.to_csv(out / "metrics" / "missing.csv")
    df.dtypes.astype(str).rename("dtype").to_csv(out / "metrics" / "dtypes.csv")
    counts = df[target].fillna("<AUSENTE>").astype(str).value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot.bar(ax=ax)
    ax.set(title="Distribuicao do target", xlabel=target, ylabel="Quantidade")
    fig.tight_layout()
    fig.savefig(out / "figures" / "target_distribution.png", dpi=150)
    plt.close(fig)
    numeric = df.drop(columns=[target]).select_dtypes(include="number")
    if not numeric.empty:
        numeric.describe().T.to_csv(out / "metrics" / "numeric_summary.csv")
        corr = numeric.corr(method="spearman")
        corr.to_csv(out / "metrics" / "spearman_correlation.csv")
        cols = list(numeric.columns[:12])
        for col in cols:
            values = numeric[col].replace([np.inf, -np.inf], np.nan).dropna()
            if values.empty:
                continue
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(values, bins=30)
            ax.set(title=f"Distribuicao: {col}", xlabel=col, ylabel="Frequencia")
            fig.tight_layout()
            fig.savefig(out / "figures" / f"numeric_{cols.index(col):02d}.png", dpi=120)
            plt.close(fig)
