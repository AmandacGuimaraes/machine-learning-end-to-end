import argparse
import json
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split
from src.data.ingestion import load_csv
from src.data.validation import validate_dataset
from src.analysis.eda import create_eda_report
from src.modeling.train import train_baseline
from src.modeling.evaluate import evaluate_binary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    df = load_csv(config["data_path"])
    target = config["target"]
    print(json.dumps(validate_dataset(df, target), ensure_ascii=False, indent=2, default=str))
    create_eda_report(df, target)
    X, y = df.drop(columns=[target]), df[target]
    positive = config.get("positive_class")
    if positive is None:
        raise ValueError("Defina positive_class explicitamente em configs/config.yaml")
    if positive not in set(y):
        raise ValueError(f"Classe positiva {positive!r} nao encontrada no target")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["test_size"], random_state=config["random_state"], stratify=y)
    model = train_baseline(X_train, y_train, config["model_path"])
    metrics = evaluate_binary(model, X_test, y_test, positive)
    path = Path("reports/metrics/baseline.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Metricas:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
