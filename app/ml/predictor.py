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

def map_input_to_features(input_dict: dict) -> dict:
    mapped = {}
    mapped["NAME_CONTRACT_TYPE"] = input_dict.get("loan_type", "Cash loans")
    mapped["CODE_GENDER"] = "M"
    mapped["FLAG_OWN_CAR"] = input_dict.get("owns_car", "N")
    mapped["FLAG_OWN_REALTY"] = input_dict.get("owns_property", "N")
    mapped["AMT_INCOME_TOTAL"] = input_dict.get("income_total", 0)
    mapped["AMT_CREDIT"] = input_dict.get("loan_amount", 0)
    mapped["AMT_ANNUITY"] = input_dict.get("loan_amount", 0) / 20
    mapped["AMT_GOODS_PRICE"] = input_dict.get("loan_amount", 0) * 0.9
    mapped["DAYS_BIRTH"] = -input_dict.get("age_years", 30) * 365
    mapped["DAYS_EMPLOYED"] = -input_dict.get("employment_years", 0) * 365
    mapped["NAME_EDUCATION_TYPE"] = input_dict.get("education", "Higher education")
    mapped["NAME_FAMILY_STATUS"] = input_dict.get("family_status", "Married")
    mapped["NAME_INCOME_TYPE"] = "Working"
    mapped["EXT_SOURCE_2"] = input_dict.get("ext_source_2", 0.5)
    mapped["EXT_SOURCE_3"] = input_dict.get("ext_source_2", 0.5) * 0.9
    return mapped

def predict_single(input_dict: dict) -> dict:
    global _xgb_model, _lgb_model, _metadata, _feature_names
    if _xgb_model is None:
        load_models()

    mapped = map_input_to_features(input_dict)
    df = pd.DataFrame([mapped])
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
        "loan_type": "Cash loans",
        "owns_car": "N",
        "owns_property": "Y",
        "income_total": 135000,
        "loan_amount": 500000,
        "age_years": 35,
        "employment_years": 4,
        "education": "Higher education",
        "family_status": "Married",
        "ext_source_2": 0.52
    })
    print("Result:", result)