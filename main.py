from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import pandas as pd
import os
import shutil
from model import BartModel, process_input_data, generate_sample_data
from pydantic import BaseModel
from typing import List
import numpy as np

# ------------------------------------------------------------------------------

app = FastAPI()
model = BartModel()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ------------------------------------------------------------------------------

# Define a Pydantic model for the input data
class ModelInput(BaseModel):
    X: List[List[float]]  # Assuming X is a 2D array (list of lists)
    y: List[float]        # Assuming y is a 1D array (list)

# ------------------------------------------------------------------------------

@app.get("/")
async def main():
    return FileResponse('static/index.html') # Serve index.html from /static

# ------------------------------------------------------------------------------

@app.post("/predict")
async def make_prediction(input_data: ModelInput):
    # print(f"Received data: X - {input_data.X}, y - {input_data.y}")
    try:
        X = np.array(input_data.X)
        y = np.array(input_data.y)
        # print(f"Data Shape: X - {X.shape}, y - {y.shape}") # Log data shape
        # print(f"Sample Data: X - {X[:5]}, y - {y[:5]}") # Log sample data
        model.fit(X, y)
        predictions = model.predict(X)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.spss'):
        raise HTTPException(status_code=400, detail="Invalid file type")

    temp_file_path = f'temp_{file.filename}'
    with open(temp_file_path, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        df = pd.read_spss(temp_file_path)
        # Return the entire DataFrame
        return {"uploadedData": df.to_dict(orient='records')}
    except Exception as e:
        os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Error reading SPSS file: {e}")
