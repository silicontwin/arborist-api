# /app/routes/predict_routes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from app.model import BartModel, generate_sample_data

class ModelInput(BaseModel):
    X: List[List[float]]
    y: List[float]

router = APIRouter()
model = BartModel()

@router.post("/predict")
async def make_prediction(input_data: ModelInput):
    try:
        X = np.array(input_data.X)
        y = np.array(input_data.y)
        model.fit(X, y)
        predictions = model.predict(X)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test_predict")
async def test_prediction():
    try:
        X, y = generate_sample_data()
        model.fit(X, y)
        predictions = model.predict(X)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
