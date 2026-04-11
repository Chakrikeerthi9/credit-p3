from fastapi import APIRouter, HTTPException
from app.models.schemas import ApplicantInput, PredictionResponse

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict(data: ApplicantInput):
    return PredictionResponse(
        risk_score=0.0,
        decision="PENDING",
        top_reasons=["Model not trained yet"],
        model_version="v1.0.0",
        processing_ms=0
    )

@router.post("/predict/batch")
async def predict_batch():
    return {"message": "Batch endpoint ready — model not trained yet"}
