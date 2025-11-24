"""
Pydantic models for air quality data.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import re


class HealthRiskLevel(str, Enum):
    """Health risk levels."""
    GOOD = "GOOD"
    MODERATE = "MODERATE"
    UNHEALTHY_FOR_SENSITIVE = "UNHEALTHY_FOR_SENSITIVE"
    UNHEALTHY = "UNHEALTHY"
    VERY_UNHEALTHY = "VERY_UNHEALTHY"
    HAZARDOUS = "HAZARDOUS"


class AreaType(str, Enum):
    """Area type classification."""
    URBAN = "urban"
    PERI_URBAN = "peri_urban"
    RURAL = "rural"
    UNKNOWN = "unknown"


class AirQualityData(BaseModel):
    """Air quality data model."""
    timestamp: datetime
    location_id: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    pm25: float = Field(..., ge=0)
    pm10: float = Field(..., ge=0)
    no2: float = Field(..., ge=0)
    co: float = Field(..., ge=0)
    o3: float = Field(..., ge=0)
    so2: float = Field(..., ge=0)
    aqi: float = Field(..., ge=0, le=500)
    primary_pollutant: Optional[str] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    wind_speed: Optional[float] = None
    wind_direction: Optional[float] = None
    pressure: Optional[float] = None
    source_api: Optional[str] = None
    quality_score: Optional[float] = None
    confidence_score: Optional[float] = None
    area_type: Optional[AreaType] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    location_id: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    prediction_horizon: int = Field(24, ge=1, le=168)  # Hours, max 1 week
    area_type: Optional[AreaType] = None


class PredictionResult(BaseModel):
    """Prediction result model."""
    location_id: str
    prediction_timestamp: datetime
    prediction_horizon: int
    predictions: Dict[str, Any]  # e.g., {"aqi": [values], "pm25": [values]}
    max_risk_level: HealthRiskLevel
    risk_timeline: Dict[str, HealthRiskLevel]  # Risk level over time
    model_version: str
    confidence_score: float = Field(..., ge=0, le=1)
    prediction_accuracy: Optional[float] = None
    source_attribution: Optional[Dict[str, Any]] = None
    ethical_explanation: Optional[str] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthAdvisory(BaseModel):
    """Health advisory model."""
    location_id: str
    timestamp: datetime
    risk_level: HealthRiskLevel
    aqi: float
    primary_concerns: List[str]
    recommendations: List[str]
    vulnerable_group_advisories: Dict[str, List[str]]
    preventive_measures: List[str]
    confidence: float = Field(..., ge=0, le=1)
    cumulative_exposure_score: Optional[float] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class WhatIfRequest(BaseModel):
    """Request model for what-if simulations."""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    scenario: Dict[str, Any] = Field(..., description="Scenario parameters")
    location_id: Optional[str] = None


class ModelMetrics(BaseModel):
    """Model performance metrics."""
    model_name: str
    mae: float
    rmse: float
    r2_score: float
    mape: float
    training_samples: int
    validation_samples: Optional[int] = None


class UserCreate(BaseModel):
    """User creation model."""
    full_name: str = Field(..., min_length=1, max_length=100)
    mobile: str = Field(..., min_length=10, max_length=20)
    email: Optional[str] = Field(None, max_length=100)
    address: Optional[str] = Field(None, max_length=500)
    age: int = Field(..., ge=1, le=120)
    chronic_conditions: Optional[str] = Field(None, max_length=500)
    is_smoker: Optional[bool] = False
    works_outdoors: Optional[bool] = False
    password: str = Field(..., min_length=6, max_length=128)
    confirm_password: str = Field(..., min_length=6, max_length=128)
    
    @validator('mobile')
    def validate_mobile(cls, v):
        # Basic mobile number validation (can be enhanced based on requirements)
        if not re.match(r'^[0-9+\-\s\(\)]{10,20}$', v):
            raise ValueError('Invalid mobile number format')
        return v
    
    @validator('email')
    def validate_email(cls, v):
        if v is not None:
            if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', v):
                raise ValueError('Invalid email format')
        return v
    
    @validator('confirm_password')
    def passwords_match(cls, v, values, **kwargs):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v


class UserLogin(BaseModel):
    """User login model."""
    identifier: str = Field(..., min_length=1, max_length=100)  # Can be mobile or email
    password: str = Field(..., min_length=1, max_length=128)


class User(BaseModel):
    """User model for response (without password)."""
    id: int
    full_name: str
    mobile: str
    email: Optional[str]
    address: Optional[str]
    age: int
    chronic_conditions: Optional[str]
    is_smoker: bool
    works_outdoors: bool
    created_at: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }