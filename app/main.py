# /app/main.py
from fastapi import FastAPI

# Absolute imports are required for pyinstaller
from app.routers.predict import router as predict
from app.routers.status import router as status
from app.routers.upload import router as upload

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(status)
app.include_router(predict)
app.include_router(upload)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
