from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


class TransactionIn(BaseModel):
    transaction_id:  str        = Field(..., description="Unique transaction identifier")
    amount:          float      = Field(..., gt=0, description="Transaction amount in USD")
    merchant_category: str      = Field(..., description="MCC code or category name")
    hour_of_day:     int        = Field(..., ge=0, le=23)
    day_of_week:     int        = Field(..., ge=0, le=6)
    is_online:       bool       = Field(default=False)
    distance_from_home_km: float = Field(default=0.0, ge=0)
    velocity_last_1h: int       = Field(default=0, ge=0, description="# transactions in last hour")
    velocity_last_24h: int      = Field(default=0, ge=0)
    account_age_days: int       = Field(default=365, ge=0)
    avg_spend_30d:   float      = Field(default=50.0, ge=0)


class PredictionOut(BaseModel):
    transaction_id:    str
    fraud_probability: float
    is_fraud:          bool
    risk_level:        str   # LOW | MEDIUM | HIGH


class TrainResponse(BaseModel):
    status:  str
    message: str


class HealthOut(BaseModel):
    status:       str
    model_loaded: bool