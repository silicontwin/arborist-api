# /app/main.py
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

# Absolute imports are required for pyinstaller
from app.routers.plot import router as plot
from app.routers.predict import router as predict
from app.routers.settings import router as settings
from app.routers.status import router as status
from app.routers.summarize import router as summarize
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
app.include_router(plot)
app.include_router(predict)
app.include_router(settings)
app.include_router(status)
app.include_router(upload)
app.include_router(summarize)


@app.get("/")
async def root():
    return RedirectResponse(url="/status")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
