"""
Pydantic schemas for request/response validation
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime


# ==================== PREDICTION SCHEMAS ====================

class PredictionRequest(BaseModel):
    """Request schema for single prediction"""
    review: str = Field(..., min_length=4, description="Review text (min 4 words)")
    rating: int = Field(..., ge=1, le=5, description="Rating 1-5")

    @validator("review")
    def validate_review(cls, v):
        """Validate review has minimum 4 words"""
        word_count = len(v.split())
        if word_count < 4:
            raise ValueError("Review must contain at least 4 words")
        if word_count > 5000:
            raise ValueError("Review cannot exceed 5000 words")
        return v


class PredictionResponse(BaseModel):
    """Response schema for prediction result"""
    id: Optional[int] = None
    review_text: str
    rating: int
    fake_score: float = Field(..., ge=0, le=1)
    trust_score: float = Field(..., ge=0, le=1)
    verdict: str
    confidence: float = Field(..., ge=0, le=100)

    # Feature breakdown
    review_length: Optional[int] = None
    word_count: Optional[int] = None
    sentiment: Optional[float] = None
    exclamation_count: Optional[int] = None
    question_count: Optional[int] = None
    caps_ratio: Optional[float] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions"""
    reviews: List[PredictionRequest] = Field(..., min_items=1, max_items=100)


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions"""
    total: int
    processed: int
    results: List[PredictionResponse]
    processing_time_ms: float


# ==================== ANALYTICS SCHEMAS ====================

class PredictionStats(BaseModel):
    """Statistics about predictions"""
    total_predictions: int
    fake_detected: int
    genuine_detected: int
    suspicious_detected: int
    fake_percentage: float
    average_confidence: float
    most_recent: Optional[datetime] = None


class ModelInfoResponse(BaseModel):
    """Response schema for model information"""
    version: str
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    trained_date: Optional[datetime] = None
    total_features: int
    threshold: float
    last_updated: datetime


class PredictionHistoryItem(BaseModel):
    """Single item in prediction history"""
    id: int
    review_text: str
    rating: int
    verdict: str
    fake_score: float
    trust_score: float
    created_at: datetime

    class Config:
        from_attributes = True


class PredictionHistoryResponse(BaseModel):
    """Response schema for prediction history"""
    total: int
    page: int
    page_size: int
    results: List[PredictionHistoryItem]


# ==================== APPEAL SCHEMAS ====================

class AppealRequest(BaseModel):
    """Request schema for appealing a verdict"""
    reason: str = Field(..., min_length=10, max_length=500)


class AppealResponse(BaseModel):
    """Response schema for appeal submission"""
    prediction_id: int
    status: str
    message: str
