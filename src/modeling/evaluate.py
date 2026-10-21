from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def evaluate_binary(model, X_test, y_test, positive_class):
    y_pred = model.predict(X_test)
    truth = (y_test == positive_class).astype(int)
    pred = (y_pred == positive_class).astype(int)
    metrics = {"accuracy": float(accuracy_score(y_test, y_pred)),
               "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
               "precision": float(precision_score(truth, pred, zero_division=0)),
               "recall": float(recall_score(truth, pred, zero_division=0)),
               "f1": float(f1_score(truth, pred, zero_division=0))}
    if len(set(truth)) == 2 and hasattr(model, "predict_proba"):
        class_index = list(model.classes_).index(positive_class)
        metrics["roc_auc"] = float(roc_auc_score(truth, model.predict_proba(X_test)[:, class_index]))
    return metrics
