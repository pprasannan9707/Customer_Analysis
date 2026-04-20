"""
routes/insights.py
GET /api/model-metrics
GET /api/feature-importance
POST /api/train (trigger retraining)
"""

from fastapi import APIRouter, BackgroundTasks
from services.model_training import load_metrics, load_feature_importance, train_all_models

router = APIRouter(prefix="/api", tags=["insights"])


@router.get("/model-metrics")
async def model_metrics():
    return load_metrics()


@router.get("/feature-importance")
async def feature_importance():
    return {"features": load_feature_importance()}


@router.post("/train")
async def trigger_training(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_all_models)
    return {"status": "Training started in background. Refresh metrics in ~60 seconds."}
