# app/routers/pyarrow.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pyarrow.dataset as ds
import os

router = APIRouter()

class FileProcessRequest(BaseModel):
    fileName: str  # Can also be a directory of CSVs
    workspacePath: str

@router.post("/pyarrow")
async def read_data(request: FileProcessRequest):
    try:
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
        
        # Convert the DataFrame to JSON
        json_data = df.to_dict(orient='records')
        
        return {"data": json_data}
    except Exception as e:
        # Log the exception for debugging
        print(f"Error processing file or directory: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process CSV file or directory: {str(e)}")
