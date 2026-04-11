from fastapi import APIRouter
from app.database import get_client

router = APIRouter()

@router.get("/audit")
async def get_audit():
    client = get_client()
    result = client.table("predictions")\
        .select("*")\
        .order("created_at", desc=True)\
        .limit(50)\
        .execute()
    return result.data

@router.get("/audit/stats")
async def get_stats():
    client = get_client()
    result = client.table("predictions").select("*").execute()
    total = len(result.data)
    approved = len([r for r in result.data if r["decision"] == "APPROVE"])
    denied = len([r for r in result.data if r["decision"] == "DENY"])
    review = len([r for r in result.data if r["decision"] == "REVIEW"])
    return {
        "total_predictions": total,
        "approved": approved,
        "denied": denied,
        "review": review
    }
