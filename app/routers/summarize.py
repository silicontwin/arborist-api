# app/routers/summarize.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pyarrow.dataset as ds
import os
import pandas as pd
import numpy as np
from app.model import BartModel
import logging
from typing import List

router = APIRouter()

class FileProcessRequest(BaseModel):
    fileName: str  # Can also be a directory of CSVs
    workspacePath: str
    selectedColumns: List[str] = []
    headTailRows: int = 20  # Number of head and tail observations to display
    action: str = "summarize" # Default action is summarize

# Instantiate the model
model = BartModel()

@router.post("/summarize")
async def read_data(request: FileProcessRequest):
    try:
        num_rows_to_display = request.headTailRows

        # Construct the full file path using the workspacePath and fileName
        file_path = os.path.join(request.workspacePath, request.fileName)

        # Check if the file or directory exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File or directory not found")

        # Create a dataset from the file or directory of CSV files
        dataset = ds.dataset(file_path, format='csv')

        # Apply filters or select columns if required
        # Example to load specific columns: dataset = dataset.to_table(columns=['col1', 'col2'])

        # Convert the dataset to a PyArrow table
        table = dataset.to_table()

        # Convert the table to a Pandas DataFrame for easier JSON serialization
        df = table.to_pandas()

        # Store the initial number of rows before removing NaN values
        initial_row_count = len(df)

        # Replace non-finite values with placeholders
        # df.replace([pd.NA, pd.NaT], 'NaN', inplace=True)
        # df.replace([float('inf'), float('-inf')], 'Infinity', inplace=True)

        # Handle missing values: remove rows with NaN values
        df_cleaned = df.dropna()

        # Calculate the number of observations removed
        observations_removed = initial_row_count - len(df_cleaned)

        response_data = {}

        if request.action == "analyze":
            # Select only numeric columns for the BART model
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                X = df_cleaned[numeric_cols].to_numpy()
                X = np.ascontiguousarray(X)
                y = X[:, -1] # Assuming y is the last column, will need to make user selectable
                X = X[:, :-1] # Subset covariates to remove outcome
                model.fit(X, y)
                predictions, lower_bound, upper_bound = model.predict(X)
                df_cleaned.insert(0, 'Posterior Average (y hat)', predictions)
                df_cleaned.insert(1, '2.75th percentile', lower_bound)
                df_cleaned.insert(2, '97.5th percentile', upper_bound) # Prepend predictions to the DataFrame
            else:
                raise HTTPException(status_code=400, detail="No numeric columns found for analysis")

        # Adjust DataFrame to include only a subset of rows based on num_rows_to_display
        if len(df_cleaned) > 2 * num_rows_to_display:
            placeholder = pd.DataFrame({col: ['...'] for col in df_cleaned.columns}, index=[0])
            df_final = pd.concat([df_cleaned.head(num_rows_to_display), placeholder, df_cleaned.tail(num_rows_to_display)], ignore_index=True)
        else:
            df_final = df_cleaned

        # Convert the DataFrame to JSON
        json_data = df_final.to_dict(orient='records')
        response_data["data"] = json_data
        response_data["wrangle"] = {"observationsRemoved": observations_removed}

        # Determine if each column is numeric and store the result in a dictionary
        is_numeric = {col: pd.api.types.is_numeric_dtype(df_cleaned[col]) for col in df_cleaned.columns}
        response_data["is_numeric"] = is_numeric

        return response_data
    except Exception as e:
        logging.error(f"Error processing file or directory: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process CSV file or directory: {str(e)}")
