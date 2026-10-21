import argparse
import json
from pathlib import Path
from src.config import load_config, validate_config
from src.data.ingestion import read_data
from src.data.validation import validate_dataframe
from src.analysis.eda import create_eda
from src.analysis.statistical_tests import associations
from src.analysis.diagnostics import diagnostics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/classification.yaml")
    args = parser.parse_args()
    cfg = validate_config(load_config(args.config))
    df = read_data(cfg["data"]["path"])
    target = cfg["data"]["target"]
    validate_dataframe(df, target)
    output = Path(cfg["outputs"]["directory"]) / "eda"
    create_eda(df, target, output)
    associations(df, target).to_csv(output / "statistical_tests.csv", index=False)
    (output / "diagnostics.json").write_text(json.dumps(diagnostics(df, target), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"EDA concluída: {output}")

if __name__ == "__main__":
    main()
