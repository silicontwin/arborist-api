from fastapi import APIRouter
from typing import Optional

router = APIRouter()

@router.get("/settings")
async def get_settings(model: str, param1: Optional[str] = None):
    return {"model": model, "param1": param1}
