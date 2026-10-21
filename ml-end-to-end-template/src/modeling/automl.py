def train_automl(train_df, target, task, output_path):
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise RuntimeError('Instale: pip install -e ".[automl]"') from exc
    problem_type = "binary" if task == "classification" and train_df[target].nunique() == 2 else ("multiclass" if task == "classification" else "regression")
    predictor = TabularPredictor(label=target, problem_type=problem_type, path=str(output_path)).fit(train_df)
    return predictor
