# app/routers/summarize.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pyarrow.dataset as ds
import os
import pandas as pd
from app.model import BartModel
import logging
from typing import List

router = APIRouter()

class FileProcessRequest(BaseModel):
    fileName: str  # Can also be a directory of CSVs
    workspacePath: str
    selectedColumns: List[str] = []
    headTailRows: int = 20  # Number of head and tail observations to display
    action: str = "summarize"

model = BartModel()

@router.post("/summarize")
async def read_data(request: FileProcessRequest):
    # Construct the full file path using the workspacePath and fileName
    file_path = os.path.join(request.workspacePath, request.fileName)
    if not os.path.exists(file_path):
        logging.error("File not found: %s", file_path)
        raise HTTPException(status_code=404, detail="File not found")

    try:
        # Create a dataset from the file or directory of CSV files
        dataset = ds.dataset(file_path, format='csv')

        # Apply filters or select columns (future Arborist feature)
        # Example to load specific columns: dataset = dataset.to_table(columns=['col1', 'col2'])

        # Convert the dataset to a PyArrow table
        table = dataset.to_table()

        # Convert the table to a Pandas DataFrame for easier JSON serialization
        df = table.to_pandas()

        # Log the received selected columns
        logging.info(f"Selected columns received: {request.selectedColumns}")

        # Ensure the DataFrame contains the columns; if not, log an error
        if request.selectedColumns and not all(col in df.columns for col in request.selectedColumns):
            missing_cols = [col for col in request.selectedColumns if col not in df.columns]
            logging.error(f"Missing columns in DataFrame: {missing_cols}")
            raise HTTPException(status_code=400, detail=f"Selected columns not found in the file: {missing_cols}")

        # Filter the DataFrame to include only the selected columns, if any are specified
        if request.selectedColumns:
            df = df[request.selectedColumns]

        # Handle non-finite values and ensure compatibility with JSON
        df.replace([pd.NA, pd.NaT], 'NaN', inplace=True)
        df.replace([float('inf'), float('-inf')], 'Infinity', inplace=True)
        df = df.applymap(lambda x: 'NaN' if pd.isna(x) else x)

        if request.action == "summarize":
            num_rows_to_display = request.headTailRows

            # Adjust DataFrame to include only a subset of rows based on num_rows_to_display
            if len(df) > 2 * num_rows_to_display:
                placeholder = pd.DataFrame({col: ['...'] for col in df.columns}, index=[num_rows_to_display])
                df_final = pd.concat([df.head(num_rows_to_display), placeholder, df.tail(num_rows_to_display)], ignore_index=True)
            else:
                df_final = df

            json_data = df_final.to_dict(orient='records')
            is_numeric = {col: pd.api.types.is_numeric_dtype(df[col]) for col in df.columns}
            return {"data": json_data, "is_numeric": is_numeric}

        elif request.action == "analyze":
            num_rows_to_display = request.headTailRows

            # Adjust DataFrame to include only a subset of rows based on num_rows_to_display
            if len(df) > 2 * num_rows_to_display:
                placeholder = pd.DataFrame({col: ['...'] for col in df.columns}, index=[num_rows_to_display])
                df_final = pd.concat([df.head(num_rows_to_display), placeholder, df.tail(num_rows_to_display)], ignore_index=True)
            else:
                df_final = df

            # Convert non-finite float values to strings
            # df_final = df_final.applymap(lambda x: 'NaN' if pd.isna(x) else x)

            # Convert the DataFrame to JSON
            json_data = df_final.to_dict(orient='records')

            # Determine if each column is numeric and store the result in a dictionary
            is_numeric = {col: pd.api.types.is_numeric_dtype(df[col]) for col in df.columns}
            return {"data": json_data, "is_numeric": is_numeric}

            # X = df.iloc[:, :-1].values
            # try:
            #     predictions = model.predict(X)
            # except Exception as e:
            #     logging.error("Model prediction failed: %s", str(e))
            #     raise HTTPException(status_code=500, detail="Model prediction failed")

            # df.insert(0, 'Predictions', predictions)
            # json_data = df.to_dict(orient='records')
            # is_numeric = {col: pd.api.types.is_numeric_dtype(df[col]) for col in df.columns}
            # return {"data": json_data, "is_numeric": is_numeric}

        else:
            logging.error("Invalid action: %s", request.action)
            raise HTTPException(status_code=400, detail="Invalid action")

    except Exception as e:
        logging.exception("An error occurred while processing the file.")
        raise HTTPException(status_code=500, detail=str(e))
