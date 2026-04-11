import shap
import numpy as np
import joblib
import pandas as pd

_xgb_explainer = None
_lgb_explainer = None

def load_explainers():
    global _xgb_explainer, _lgb_explainer
    xgb_model = joblib.load("models/xgb_v1.pkl")
    lgb_model = joblib.load("models/lgbm_v1.pkl")
    _xgb_explainer = shap.TreeExplainer(xgb_model)
    _lgb_explainer = shap.TreeExplainer(lgb_model)
    print("SHAP explainers loaded")

def get_top_reasons(X_row: pd.DataFrame, n: int = 3) -> list:
    global _xgb_explainer
    if _xgb_explainer is None:
        load_explainers()
    shap_values = _xgb_explainer.shap_values(X_row)
    feature_names = X_row.columns.tolist()
    shap_series = pd.Series(
        np.abs(shap_values[0]),
        index=feature_names
    ).sort_values(ascending=False)
    top_features = shap_series.head(n).index.tolist()
    reasons = []
    for feat in top_features:
        val = X_row[feat].values[0]
        shap_val = shap_values[0][feature_names.index(feat)]
        direction = "increases" if shap_val > 0 else "decreases"
        reasons.append(f"{feat} = {val:.2f} {direction} risk")
    return reasons

if __name__ == "__main__":
    from app.ml.feature_store import run_pipeline
    X, y = run_pipeline("data/application_train.csv")
    load_explainers()
    reasons = get_top_reasons(X.iloc[[0]])
    print("Top reasons:", reasons)
