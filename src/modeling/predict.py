import joblib

def predict(model_path, features):
    model = joblib.load(model_path)  # Carregue apenas artefatos confiaveis.
    return model.predict(features)
