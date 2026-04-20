"""
routes/summary.py
GET /api/summary
GET /api/customers
GET /api/segment
POST /api/segment-summary (AI)
POST /api/strategy (AI campaign strategy)
"""

from fastapi import APIRouter, Query
from typing import Optional
import asyncio

from services.explain_service import get_overview_summary, get_segment_stats, get_all_customers
from services.hf_service import generate_segment_summary, generate_campaign_strategy

router = APIRouter(prefix="/api", tags=["summary"])


@router.get("/summary")
async def summary():
    return get_overview_summary()


@router.get("/customers")
async def customers(limit: int = Query(default=500, le=2500)):
    return {"customers": get_all_customers(limit=limit)}


@router.get("/segment")
async def segment(
    education: Optional[str] = None,
    marital_status: Optional[str] = None,
    income_band: Optional[str] = None,
    tenure_band: Optional[str] = None,
    children: Optional[int] = None,
):
    return get_segment_stats(
        education=education,
        marital_status=marital_status,
        income_band=income_band,
        tenure_band=tenure_band,
        children=children,
    )


@router.post("/segment-summary")
async def segment_summary_ai(body: dict):
    segment_name = body.get("segment_name", "Custom Segment")
    stats = body.get("stats", {})
    text = await generate_segment_summary(segment_name, stats)
    return {"summary": text}


@router.post("/strategy")
async def campaign_strategy(body: dict):
    overall = body.get("overall", {})
    segments = body.get("segments", [])
    text = await generate_campaign_strategy(overall, segments)
    return {"strategy": text}
