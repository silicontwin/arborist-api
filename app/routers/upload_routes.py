from fastapi import APIRouter, HTTPException, UploadFile, File
import pandas as pd
import os
import shutil

router = APIRouter()

@router.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    temp_file_path = f'temp_{file.filename}'
    with open(temp_file_path, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        if file.filename.endswith('.spss'):
            df = pd.read_spss(temp_file_path)
            return {"uploadedData": df.to_dict(orient='records')}
        else:
            with open(temp_file_path, 'r') as f:
                file_content = f.read()
            return {"uploadedData": file_content}
    except Exception as e:
        os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")
