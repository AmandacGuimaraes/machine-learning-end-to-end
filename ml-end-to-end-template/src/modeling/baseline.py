from sklearn.dummy import DummyClassifier, DummyRegressor

def baseline_model(task):
    return DummyClassifier(strategy="most_frequent") if task == "classification" else DummyRegressor(strategy="mean")
