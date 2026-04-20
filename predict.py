"""
routes/predict.py
POST /api/predict-customer
POST /api/predict-batch
POST /api/what-if
POST /api/ai-explanation
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Optional
import asyncio

from services.predict_service import predict_customer, predict_batch
from services.hf_service import explain_prediction, recommend_action, explain_whatif

router = APIRouter(prefix="/api", tags=["predict"])


class CustomerInput(BaseModel):
    customer: dict[str, Any]


class BatchInput(BaseModel):
    customers: list[dict[str, Any]]


class ExplainInput(BaseModel):
    customer: dict[str, Any]
    prediction_result: Optional[dict[str, Any]] = None


class WhatIfInput(BaseModel):
    original_customer: dict[str, Any]
    modified_customer: dict[str, Any]
    changed_fields: Optional[dict[str, Any]] = {}


@router.post("/predict-customer")
async def predict_single(body: CustomerInput):
    try:
        result = predict_customer(body.customer)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-batch")
async def predict_batch_endpoint(body: BatchInput):
    try:
        results = predict_batch(body.customers)
        return {"predictions": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ai-explanation")
async def ai_explanation(body: ExplainInput):
    """Generate AI explanation + recommended action for a single customer."""
    try:
        pred = body.prediction_result
        if pred is None:
            pred = predict_customer(body.customer)

        explanation, action = await asyncio.gather(
            explain_prediction(
                customer_profile=body.customer,
                probability=pred["probability"],
                top_features=pred.get("top_features", []),
                risk_label=pred.get("risk_label", ""),
            ),
            recommend_action(
                customer_profile=body.customer,
                probability=pred["probability"],
                top_features=pred.get("top_features", []),
            ),
        )

        return {
            "explanation": explanation,
            "recommended_action": action,
            "probability": pred["probability"],
            "risk_label": pred["risk_label"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/what-if")
async def what_if(body: WhatIfInput):
    """Compare original vs modified customer prediction."""
    try:
        original_result = predict_customer(body.original_customer)
        new_result = predict_customer(body.modified_customer)

        ai_text = await explain_whatif(
            original_prob=original_result["probability"],
            new_prob=new_result["probability"],
            changed_fields=body.changed_fields or {},
            top_features=new_result.get("top_features", []),
        )

        return {
            "original": original_result,
            "modified": new_result,
            "delta": round(new_result["probability"] - original_result["probability"], 4),
            "ai_analysis": ai_text,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
