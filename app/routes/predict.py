import time
import io
import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File
from app.models.schemas import ApplicantInput, PredictionResponse, BatchResponse
from app.ml.predictor import predict_single
from app.database import get_client
from app.config import settings

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict(data: ApplicantInput):
    start = time.time()
    try:
        result = predict_single(data.dict())
        processing_ms = int((time.time() - start) * 1000)

        client = get_client()
        client.table("predictions").insert({
            "model_version": settings.MODEL_VERSION,
            "age_years": data.age_years,
            "income_bracket": "high" if data.income_total > 100000 else "low",
            "loan_amount": data.loan_amount,
            "loan_type": data.loan_type,
            "risk_score": result["risk_score"],
            "decision": result["decision"],
            "top_reason_1": result["top_reasons"][0] if len(result["top_reasons"]) > 0 else "",
            "top_reason_2": result["top_reasons"][1] if len(result["top_reasons"]) > 1 else "",
            "top_reason_3": result["top_reasons"][2] if len(result["top_reasons"]) > 2 else "",
            "processing_ms": processing_ms
        }).execute()

        return PredictionResponse(
            risk_score=result["risk_score"],
            decision=result["decision"],
            top_reasons=result["top_reasons"],
            model_version=result["model_version"],
            processing_ms=processing_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files accepted")

    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    if len(df) > settings.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Max {settings.MAX_BATCH_SIZE} rows allowed"
        )

    results = []
    for _, row in df.iterrows():
        try:
            result = predict_single(row.to_dict())
            results.append({
                "risk_score": result["risk_score"],
                "decision": result["decision"],
                "top_reason_1": result["top_reasons"][0] if result["top_reasons"] else ""
            })
        except Exception:
            results.append({
                "risk_score": -1,
                "decision": "ERROR",
                "top_reason_1": "Processing failed"
            })

    approved = len([r for r in results if r["decision"] == "APPROVE"])
    denied = len([r for r in results if r["decision"] == "DENY"])
    review = len([r for r in results if r["decision"] == "REVIEW"])

    return BatchResponse(
        total=len(results),
        approved=approved,
        denied=denied,
        review=review,
        results=results
    )
