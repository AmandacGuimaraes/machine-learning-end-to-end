import argparse
from pathlib import Path
from src.data.ingestion import read_data
from src.monitoring.drift import numeric_drift

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--output", default="artifacts/drift.csv")
    args = parser.parse_args()
    report = numeric_drift(read_data(args.reference), read_data(args.current))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)
    print(f"Relatório: {args.output}")

if __name__ == "__main__":
    main()
