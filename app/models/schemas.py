from pydantic import BaseModel, validator
from typing import Optional, List

class ApplicantInput(BaseModel):
    loan_type: str
    age_years: int
    income_total: float
    loan_amount: float
    employment_years: float
    education: str
    family_status: str
    owns_property: str
    owns_car: str
    ext_source_2: Optional[float] = 0.5

    @validator("age_years")
    def age_must_be_valid(cls, v):
        if v < 18 or v > 80:
            raise ValueError("Age must be between 18 and 80")
        return v

    @validator("income_total")
    def income_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Income must be positive")
        return v

    @validator("employment_years")
    def employment_must_be_valid(cls, v, values):
        if "age_years" in values and v > values["age_years"] - 18:
            raise ValueError("Employment years cannot exceed age minus 18")
        return v

class PredictionResponse(BaseModel):
    risk_score: float
    decision: str
    top_reasons: List[str]
    model_version: str
    processing_ms: int
    cached: bool = False

class BatchResponse(BaseModel):
    total: int
    approved: int
    denied: int
    review: int
    results: list