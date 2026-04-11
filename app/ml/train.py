import numpy as np
import joblib
import json
import os
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from scipy import stats
import xgboost as xgb
import lightgbm as lgb
from app.ml.feature_store import run_pipeline

DATA_PATH = "data/application_train.csv"
MODELS_DIR = "models"

def ks_statistic(y_true, y_pred):
    pos_scores = y_pred[y_true == 1]
    neg_scores = y_pred[y_true == 0]
    ks, _ = stats.ks_2samp(pos_scores, neg_scores)
    return ks

def train():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading and processing data...")
    X, y = run_pipeline(DATA_PATH)

    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="auc",
        verbosity=0
    )
    xgb_scores = cross_val_score(
        xgb_model, X, y, cv=cv,
        scoring="roc_auc", n_jobs=-1
    )
    xgb_auc = xgb_scores.mean()
    print(f"XGBoost AUC: {xgb_auc:.4f} ± {xgb_scores.std():.4f}")

    print("\nTraining LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=-1
    )
    lgb_scores = cross_val_score(
        lgb_model, X, y, cv=cv,
        scoring="roc_auc", n_jobs=-1
    )
    lgb_auc = lgb_scores.mean()
    print(f"LightGBM AUC: {lgb_auc:.4f} ± {lgb_scores.std():.4f}")

    total_auc = xgb_auc + lgb_auc
    xgb_weight = xgb_auc / total_auc
    lgb_weight = lgb_auc / total_auc
    print(f"\nEnsemble weights: XGB={xgb_weight:.3f} LGB={lgb_weight:.3f}")

    print("\nFitting final models on full data...")
    xgb_model.fit(X, y)
    lgb_model.fit(X, y)

    xgb_pred = xgb_model.predict_proba(X)[:, 1]
    lgb_pred = lgb_model.predict_proba(X)[:, 1]
    ensemble_pred = (xgb_pred * xgb_weight) + (lgb_pred * lgb_weight)

    final_auc = roc_auc_score(y, ensemble_pred)
    ks = ks_statistic(y.values, ensemble_pred)
    gini = 2 * final_auc - 1

    print(f"\n=== Final Metrics ===")
    print(f"AUC-ROC:  {final_auc:.4f}")
    print(f"KS:       {ks:.4f}")
    print(f"Gini:     {gini:.4f}")

    joblib.dump(xgb_model, f"{MODELS_DIR}/xgb_v1.pkl")
    joblib.dump(lgb_model, f"{MODELS_DIR}/lgbm_v1.pkl")

    metadata = {
        "model_version": "v1.0.0",
        "trained_at": datetime.now().isoformat(),
        "xgb_cv_auc": round(xgb_auc, 4),
        "lgb_cv_auc": round(lgb_auc, 4),
        "ensemble_auc": round(final_auc, 4),
        "ks_statistic": round(ks, 4),
        "gini": round(gini, 4),
        "xgb_weight": round(xgb_weight, 4),
        "lgb_weight": round(lgb_weight, 4),
        "n_features": X.shape[1],
        "feature_names": list(X.columns),
        "scale_pos_weight": round(scale_pos_weight, 2)
    }

    with open(f"{MODELS_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModels saved to {MODELS_DIR}/")
    return metadata

if __name__ == "__main__":
    train()
