# app/routers/summarize.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pyarrow.dataset as ds
import os
import pandas as pd

router = APIRouter()

class FileProcessRequest(BaseModel):
    fileName: str  # Can also be a directory of CSVs
    workspacePath: str
    headTailRows: int = 20  # Number of head and tail observations to display

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

        # Apply filters or select columns (future Arborist feature)
        # Example to load specific columns: dataset = dataset.to_table(columns=['col1', 'col2'])

        # Convert the dataset to a PyArrow table
        table = dataset.to_table()

        # Convert the table to a Pandas DataFrame for easier JSON serialization
        df = table.to_pandas()

        # Replace non-finite values with placeholders
        df.replace([pd.NA, pd.NaT], 'NaN', inplace=True)
        df.replace([float('inf'), float('-inf')], 'Infinity', inplace=True)

        # Adjust DataFrame to include only a subset of rows based on num_rows_to_display
        if len(df) > 2 * num_rows_to_display:
            placeholder = pd.DataFrame({col: ['...'] for col in df.columns}, index=[0])
            df_final = pd.concat([df.head(num_rows_to_display), placeholder, df.tail(num_rows_to_display)], ignore_index=True)
        else:
            df_final = df

        # Convert non-finite float values to strings
        df_final = df_final.applymap(lambda x: 'NaN' if pd.isna(x) else x)

        # Convert the DataFrame to JSON
        json_data = df_final.to_dict(orient='records')

        # Determine if each column is numeric and store the result in a dictionary
        is_numeric = {col: pd.api.types.is_numeric_dtype(df[col]) for col in df.columns}

        return {"data": json_data, "is_numeric": is_numeric}
    except Exception as e:
        print(f"Error processing file or directory: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process CSV file or directory: {str(e)}")
