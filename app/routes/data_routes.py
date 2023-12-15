from fastapi import APIRouter

router = APIRouter()

@router.get("/data")
def read_data():
    return {"status": "active"}
