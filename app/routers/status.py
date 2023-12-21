from fastapi import APIRouter, Response, status
from typing import Dict

router = APIRouter()

@router.get(
    "/status", 
    summary="Check that the API is online [Arborist App]",
    description="This endpoint checks if the API is online and reachable, and is primarily used by the Arborist App to determine if the API is available when starting the app.",
    response_description="The status of the API",
    response_model=Dict[str, str],
    responses={
        200: {"description": "API is online"},
        500: {"description": "Internal server error"}
    }
)
def read_data():
    """
    Read the status of the API.

    This can be used to check the health or availability of the API.
    Returns a JSON object with the current status.
    """
    return {"api": "online"}
