# main.py
from fastapi import FastAPI, HTTPException
from model import BartModel, process_input_data, generate_sample_data
from pydantic import BaseModel
from typing import List
import numpy as np

app = FastAPI()
model = BartModel()

@app.get("/")
def read_root():
    return {"message": "Hello TxBSPI!"}

# Define a Pydantic model for the input data
class ModelInput(BaseModel):
    X: List[List[float]]  # Assuming X is a 2D array (list of lists)
    y: List[float]        # Assuming y is a 1D array (list)

@app.post("/predict")
async def make_prediction(input_data: ModelInput):
    try:
        X = np.array(input_data.X)
        y = np.array(input_data.y)
        print(f"Data Shape: X - {X.shape}, y - {y.shape}") # Log data shape
        print(f"Sample Data: X - {X[:5]}, y - {y[:5]}") # Log sample data
        model.fit(X, y)
        predictions = model.predict(X)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/test_predict")
async def test_prediction():
    try:
        X, y = generate_sample_data() # Use the function to generate data
        print(f"Generated Data Shape: X - {X.shape}, y - {y.shape}")
        model.fit(X, y)
        predictions = model.predict(X)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        print(f"Error during test prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))