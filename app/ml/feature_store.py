import pandas as pd
import numpy as np

TARGET = "TARGET"
ID_COL = "SK_ID_CURR"

CATEGORICAL_COLS = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
    "ORGANIZATION_TYPE"
]

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"Default rate: {df[TARGET].mean():.1%}")
    return df

def drop_high_null_cols(df: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
    null_ratio = df.isnull().mean()
    drop_cols = null_ratio[null_ratio > threshold].index.tolist()
    drop_cols = [c for c in drop_cols if c not in [TARGET, ID_COL]]
    df = df.drop(columns=drop_cols)
    print(f"Dropped {len(drop_cols)} columns with >{threshold*100:.0f}% nulls")
    return df

def fix_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED_ANOMALY"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, 0)
    if "AMT_INCOME_TOTAL" in df.columns:
        cap = df["AMT_INCOME_TOTAL"].quantile(0.99)
        df["AMT_INCOME_TOTAL"] = df["AMT_INCOME_TOTAL"].clip(upper=cap)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
    if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    if "AMT_CREDIT" in df.columns and "AMT_GOODS_PRICE" in df.columns:
        df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"] + 1)
    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365).astype(int)
    if "DAYS_EMPLOYED" in df.columns:
        df["EMPLOYMENT_YEARS"] = (-df["DAYS_EMPLOYED"] / 365).clip(lower=0)
    return df

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes
    return df

def impute_nulls(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if col not in [TARGET, ID_COL]:
            df[col] = df[col].fillna(df[col].median())
    return df

def get_features_and_target(df: pd.DataFrame):
    drop = [TARGET, ID_COL]
    drop = [c for c in drop if c in df.columns]
    X = df.drop(columns=drop)
    y = df[TARGET] if TARGET in df.columns else None
    return X, y

def run_pipeline(path: str):
    df = load_data(path)
    df = drop_high_null_cols(df)
    df = fix_anomalies(df)
    df = engineer_features(df)
    df = encode_categoricals(df)
    df = impute_nulls(df)
    X, y = get_features_and_target(df)
    print(f"Final features: {X.shape[1]}")
    print(f"Feature names: {list(X.columns[:10])}...")
    return X, y

if __name__ == "__main__":
    X, y = run_pipeline("data/application_train.csv")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Class balance: {y.mean():.1%} defaults")
