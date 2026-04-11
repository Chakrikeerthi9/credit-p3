from fastapi import APIRouter
from app.config import settings
from app.database import get_client

router = APIRouter()

@router.get("/health")
async def health():
    try:
        client = get_client()
        client.table("predictions").select("id").limit(1).execute()
        db_status = "ok"
    except Exception:
        db_status = "unavailable"
    return {
        "status": "ok" if db_status == "ok" else "degraded",
        "database": db_status,
        "model_version": settings.MODEL_VERSION
    }

@router.get("/model/info")
async def model_info():
    return {
        "model_version": settings.MODEL_VERSION,
        "models": ["XGBoost", "LightGBM"],
        "ensemble": "weighted average",
        "features": 58,
        "target_auc": 0.75,
        "status": "not trained yet"
    }
