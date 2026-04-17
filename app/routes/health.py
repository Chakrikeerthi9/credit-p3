from fastapi import APIRouter
from app.config import settings
from app.database import get_client
import json
import os

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
    if os.path.exists("models/metadata.json"):
        with open("models/metadata.json") as f:
            meta = json.load(f)
        return {
            "model_version": meta["model_version"],
            "models": ["XGBoost", "LightGBM"],
            "ensemble": "weighted average",
            "features": meta["n_features"],
            "auc_roc": meta["ensemble_auc"],
            "ks_statistic": meta["ks_statistic"],
            "gini": meta["gini"],
            "xgb_cv_auc": meta["xgb_cv_auc"],
            "lgb_cv_auc": meta["lgb_cv_auc"],
            "status": "trained"
        }
    return {
        "model_version": settings.MODEL_VERSION,
        "status": "not trained yet"
    }