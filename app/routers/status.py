from fastapi import APIRouter

router = APIRouter()

@router.get("/status")
def read_data():
    return {"api": "online"}
