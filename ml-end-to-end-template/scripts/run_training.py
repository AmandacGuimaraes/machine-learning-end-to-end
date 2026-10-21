import argparse
import json
from pathlib import Path
import joblib
from src.config import load_config, validate_config
from src.data.ingestion import read_data
from src.data.validation import validate_dataframe
from src.features.splitting import split_data
from src.modeling.train import make_pipeline
from src.modeling.baseline import baseline_model
from src.modeling.optimization import optimize_random_forest
from src.evaluation.metrics import calculate_metrics
from src.evaluation.robustness import bootstrap_metric_ci
from src.evaluation.explainability import permutation_importances

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/classification.yaml")
    args = parser.parse_args()
    cfg = validate_config(load_config(args.config))
    task = cfg["project"]["task"]
    df = read_data(cfg["data"]["path"])
    validate_dataframe(df, cfg["data"]["target"])
    X_train, X_test, y_train, y_test = split_data(df, cfg)
    out = Path(cfg["outputs"]["directory"])
    out.mkdir(parents=True, exist_ok=True)
    baseline = make_pipeline(X_train, task, cfg["model"]["algorithm"], cfg["validation"]["random_state"])
    from sklearn.pipeline import Pipeline
    baseline_dummy = Pipeline([("preprocess", baseline.named_steps["preprocess"]), ("model", baseline_model(task))])
    baseline_dummy.fit(X_train, y_train)
    baseline_metrics = calculate_metrics(baseline_dummy, X_test, y_test, task)
    if cfg["model"].get("optimize"):
        if cfg["model"]["algorithm"] != "random_forest":
            raise ValueError("Otimização integrada disponível apenas para random_forest")
        model, best_params = optimize_random_forest(baseline, X_train, y_train, task)
    else:
        model = baseline.fit(X_train, y_train)
        best_params = {}
    metrics = calculate_metrics(model, X_test, y_test, task)
    report = {"baseline": baseline_metrics, "model": metrics, "bootstrap_ci": bootstrap_metric_ci(model, X_test, y_test, task), "best_params": best_params}
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    permutation_importances(model, X_test, y_test, task).to_csv(out / "feature_importance.csv", index=False)
    joblib.dump(model, out / "model.joblib")
    print(json.dumps(report, indent=2))
    print(f"Modelo salvo em {out / 'model.joblib'}")
    if cfg["model"].get("automl"):
        from src.modeling.automl import train_automl
        train_df = X_train.copy()
        train_df[cfg["data"]["target"]] = y_train
        predictor = train_automl(train_df, cfg["data"]["target"], task, out / "autogluon")
        print("AutoGluon:", predictor.evaluate(X_test.assign(**{cfg["data"]["target"]: y_test})))

if __name__ == "__main__":
    main()
