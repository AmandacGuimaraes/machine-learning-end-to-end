import os
from functools import lru_cache
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="ML End-to-End API", version="0.1.0")

class PredictionRequest(BaseModel):
    records: list[dict]

@lru_cache
def load_model():
    path = Path(os.getenv("MODEL_DIR", "artifacts")) / "model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {path}")
    return joblib.load(path)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    try:
        load_model()
        return {"status": "ready"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

@app.post("/predict")
def predict(request: PredictionRequest):
    if not request.records:
        raise HTTPException(status_code=422, detail="Envie ao menos um registro")
    try:
        model = load_model()
        frame = pd.DataFrame(request.records)
        predictions = model.predict(frame).tolist()
        return {"predictions": [p.item() if hasattr(p, "item") else p for p in predictions]}
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
