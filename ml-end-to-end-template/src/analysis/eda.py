from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def create_eda(df, target, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df.describe(include="all").T.to_csv(outdir / "describe.csv")
    pd.DataFrame({"missing": df.isna().sum(), "missing_pct": df.isna().mean() * 100}).to_csv(outdir / "missing.csv")
    df[target].value_counts(dropna=False).to_csv(outdir / "target_distribution.csv")
    ax = df[target].value_counts(dropna=False).plot(kind="bar", title="Distribuição do target")
    ax.set_xlabel(target)
    plt.tight_layout()
    plt.savefig(outdir / "target.png")
    plt.close()
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] > 1:
        numeric.corr(method="spearman").to_csv(outdir / "spearman.csv")
