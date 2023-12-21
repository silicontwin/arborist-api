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

@router.post(
    "/predict",
    summary="Make a Prediction",
    description="Receives input data for the model and returns the model's predictions.",
    response_description="Predictions based on the input data",
    responses={
        200: {"description": "Prediction successful"},
        500: {"description": "Internal server error due to model or data issues"}
    }
)
async def make_prediction(input_data: ModelInput):
    """
    Make a prediction based on the provided input data.

    This endpoint accepts a list of features (X) and their corresponding outputs (y) to fit the model,
    and then uses the fitted model to make predictions.

    Args:
        input_data (ModelInput): Input data for the model, including features and outputs.

    Returns:
        dict: A dictionary with a key 'predictions' containing the list of predicted values.
    """
    try:
        X = np.array(input_data.X)
        y = np.array(input_data.y)
        model.fit(X, y)
        predictions = model.predict(X)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/test_predict",
    summary="Test Prediction with Sample Data",
    description="Generates sample data, fits the model, and returns predictions for testing purposes.",
    response_description="Predictions based on the sample data",
    responses={
        200: {"description": "Test prediction successful"},
        500: {"description": "Internal server error due to model or data issues"}
    }
)
async def test_prediction():
    """
    Test the prediction capability of the model using generated sample data.

    This endpoint generates sample data, fits the model with this data, 
    and then uses the model to make predictions on the same data.

    Returns:
        dict: A dictionary with a key 'predictions' containing the list of predicted values from the sample data.
    """
    try:
        X, y = generate_sample_data()
        model.fit(X, y)
        predictions = model.predict(X)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

