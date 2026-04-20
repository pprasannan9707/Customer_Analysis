"""
app.py
FastAPI application entry point for Campaign Intelligence Platform.
Starts model training on startup if no saved model exists.
"""

import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from routes.summary import router as summary_router
from routes.predict import router as predict_router
from routes.insights import router as insights_router
from services.model_training import BEST_MODEL_PATH, train_all_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Train model on startup if no saved model found."""
    if not os.path.exists(BEST_MODEL_PATH):
        print("[Startup] No trained model found. Training now (this takes ~60s)...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, train_all_models)
        print("[Startup] Model training complete.")
    else:
        print("[Startup] Pre-trained model found. Ready.")
    yield


app = FastAPI(
    title="Campaign Intelligence Platform",
    description="GenAI-powered customer response prediction and analytics",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(summary_router)
app.include_router(predict_router)
app.include_router(insights_router)


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "model_ready": os.path.exists(BEST_MODEL_PATH),
        "version": "1.0.0",
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
