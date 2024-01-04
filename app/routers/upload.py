from fastapi import APIRouter, HTTPException, UploadFile, File
import pandas as pd
import os
import shutil

router = APIRouter()

@router.post(
    "/upload",
    summary="Upload and process a file",
    description="""
        Uploads a file for processing. 
        This endpoint currently supports '.spss' files for processing with pandas.
        Other file types will be read as plain text.
        The file is temporarily stored, processed, and then the temporary file is deleted.
    """,
    responses={
        200: {"description": "File uploaded and processed successfully"},
        400: {"description": "Invalid file format or content"},
        500: {"description": "Internal server error during file processing"}
    }
)
async def create_upload_file(file: UploadFile = File(...)):
    """
    Handle file upload and processing.

    The function supports the '.spss' file format, which will be processed using pandas.
    Other file formats are returned as plain text content.
    After processing, the file is deleted from the server.

    Args:
        file (UploadFile): The file to be uploaded and processed.

    Returns:
        dict: Contains the processed data from the file.
    
    Raises:
        HTTPException: If there is an error in processing the file.

    Example return for '.spss' file:
    {
        "uploadedData": [{...}]  # Data in record format
    }

    Example return for other file types:
    {
        "uploadedData": "file content as plain text"
    }
    """
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
