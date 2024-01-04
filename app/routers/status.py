from fastapi import APIRouter, Response, status
from typing import Dict

router = APIRouter()

@router.get(
    "/status",
    summary="Check that the API is online [Arborist App]",
    description="""
        This endpoint checks if the API is online and reachable. 
        It's primarily used by the Arborist App to determine if the API is available upon startup.
        Useful for health checks and monitoring the availability of the API services.
        No authentication is required for this endpoint.
    """,
    response_description="JSON object indicating the status of the API",
    response_model=Dict[str, str],
    responses={
        200: {"description": "API is online and reachable"},
        500: {"description": "Internal server error, indicating a problem within the server"}
    }
)

def read_data():
    """
    Read the status of the API.

    Use this endpoint to check the health or availability of the API.
    Returns a JSON object with the key 'api' and value 'online' indicating the API's status.
    
    Example response:
    {
        "api": "online"
    }
    """
    return {"api": "online"}
