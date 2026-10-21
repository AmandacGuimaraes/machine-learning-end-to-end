from sklearn.model_selection import RandomizedSearchCV

def optimize_random_forest(pipeline, X, y, task, random_state=42):
    scoring = "balanced_accuracy" if task == "classification" else "neg_mean_absolute_error"
    search = RandomizedSearchCV(
        pipeline,
        {"model__max_depth": [None, 4, 8, 16], "model__min_samples_leaf": [1, 2, 5, 10]},
        n_iter=8, cv=3, scoring=scoring, random_state=random_state, n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_
