from fastapi import APIRouter, HTTPException
import pyarrow.dataset as ds
import os

router = APIRouter()

@router.get("/pyarrow")
async def read_data():
    static_dir = 'static'
    csv_file_path = os.path.join(static_dir, 'test.csv') # Can also be a directory of CSVs

    try:
        # Create a dataset from the CSV file(s)
        dataset = ds.dataset(csv_file_path, format='csv')

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
        raise HTTPException(status_code=500, detail=f"Failed to read CSV file: {str(e)}")
