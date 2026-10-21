import argparse
from pathlib import Path
import pandas as pd
from sklearn.datasets import make_classification, make_regression

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--rows", type=int, default=1000)
    args = parser.parse_args()
    if args.task == "classification":
        X, y = make_classification(n_samples=args.rows, n_features=10, n_informative=6, random_state=42)
    else:
        X, y = make_regression(n_samples=args.rows, n_features=10, n_informative=6, noise=15, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    df["target"] = y
    path = Path(f"data/sample/{args.task}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Gerado: {path}")

if __name__ == "__main__":
    main()
