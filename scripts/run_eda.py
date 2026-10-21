import argparse
import json
from pathlib import Path
import pandas as pd
import yaml
from src.data.ingestion import load_csv
from src.analysis.eda import create_eda_report
from src.analysis.data_diagnostics import diagnose
from src.analysis.statistical_tests import categorical_association, numeric_association, benjamini_hochberg

def main():
    parser = argparse.ArgumentParser(description="EDA e testes estatisticos")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", default="reports")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    target = config["target"]
    df = load_csv(config["data_path"])
    output = Path(args.output)
    (output / "metrics").mkdir(parents=True, exist_ok=True)
    diagnostic = diagnose(df, target)
    (output / "metrics" / "diagnostics.json").write_text(json.dumps(diagnostic, indent=2, ensure_ascii=False), encoding="utf-8")
    create_eda_report(df, target, str(output))
    # Os testes sao exploratorios; evite usar o conjunto de teste final para selecionar features.
    rows = []
    for col in df.columns:
        if col == target or df[col].nunique(dropna=True) < 2:
            continue
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                result = numeric_association(df, col, target)
            elif df[col].nunique(dropna=True) <= 30:
                result = categorical_association(df, col, target)
            else:
                continue
            rows.append({"feature": col, **result})
        except (ValueError, TypeError):
            continue
    if rows:
        results = pd.DataFrame(rows)
        results["p_adjusted_bh"] = benjamini_hochberg(results["p_value"].tolist())
        results.sort_values("p_adjusted_bh").to_csv(output / "metrics" / "statistical_tests.csv", index=False)
    print(f"EDA concluida. Resultados: {output.resolve()}")
    print(f"Linhas: {diagnostic['rows']} | Colunas: {diagnostic['columns']} | Duplicatas: {diagnostic['duplicate_rows']}")
    print("Aviso: associacao estatistica nao comprova causalidade nem poder preditivo.")

if __name__ == "__main__":
    main()
