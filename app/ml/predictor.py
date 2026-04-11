import joblib
import json
import numpy as np
import pandas as pd
from app.ml.feature_store import (
    fix_anomalies, engineer_features,
    encode_categoricals, impute_nulls
)
from app.ml.explainer import get_top_reasons

_xgb_model = None
_lgb_model = None
_metadata = None
_feature_names = None

def load_models():
    global _xgb_model, _lgb_model, _metadata, _feature_names
    _xgb_model = joblib.load("models/xgb_v1.pkl")
    _lgb_model = joblib.load("models/lgbm_v1.pkl")
    with open("models/metadata.json") as f:
        _metadata = json.load(f)
    _feature_names = _metadata["feature_names"]
    print("Models loaded")

def get_decision(score: float) -> str:
    if score < 0.3:
        return "APPROVE"
    elif score < 0.6:
        return "REVIEW"
    else:
        return "DENY"

def predict_single(input_dict: dict) -> dict:
    global _xgb_model, _lgb_model, _metadata, _feature_names
    if _xgb_model is None:
        load_models()

    df = pd.DataFrame([input_dict])
    df = fix_anomalies(df)
    df = engineer_features(df)
    df = encode_categoricals(df)
    df = impute_nulls(df)

    missing_cols = {col: 0 for col in _feature_names if col not in df.columns}
    df = pd.concat([df, pd.DataFrame([missing_cols])], axis=1)
    df = df[_feature_names].copy()

    xgb_pred = _xgb_model.predict_proba(df)[:, 1][0]
    lgb_pred = _lgb_model.predict_proba(df)[:, 1][0]

    xgb_weight = _metadata["xgb_weight"]
    lgb_weight = _metadata["lgb_weight"]
    score = float(xgb_pred * xgb_weight + lgb_pred * lgb_weight)

    reasons = get_top_reasons(df)
    decision = get_decision(score)

    return {
        "risk_score": round(score, 4),
        "decision": decision,
        "top_reasons": reasons,
        "model_version": _metadata["model_version"]
    }

if __name__ == "__main__":
    result = predict_single({
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": "M",
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "Y",
        "CNT_CHILDREN": 0,
        "AMT_INCOME_TOTAL": 135000,
        "AMT_CREDIT": 500000,
        "AMT_ANNUITY": 25000,
        "AMT_GOODS_PRICE": 450000,
        "DAYS_BIRTH": -12000,
        "DAYS_EMPLOYED": -2000,
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Married",
        "NAME_INCOME_TYPE": "Working",
        "EXT_SOURCE_2": 0.52,
        "EXT_SOURCE_3": 0.48
    })
    print("Result:", result)
