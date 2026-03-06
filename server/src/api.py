"""
FastAPI application for Fake Review Detection
"""
from fastapi import FastAPI, Depends, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, Index
from sqlalchemy.engine import Engine
from typing import List, Optional
import time
import logging
from datetime import datetime
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import storage rate limiter
from .rate_limiter import rate_limit_middleware

from .database import get_db, init_db
from .models import Prediction, ModelMetadata
from .schemas import (
    PredictionRequest, PredictionResponse, BatchPredictionRequest,
    BatchPredictionResponse, PredictionStats, ModelInfoResponse,
    PredictionHistoryResponse, PredictionHistoryItem, AppealRequest,
    AppealResponse
)
from .predict import load_models, predict as ml_predict, is_model_loaded
from .utils import init_nltk
from . import config


# Store startup time for uptime tracking
_startup_time = None


# Initialize FastAPI app
app = FastAPI(
    title="Fake Review Detection API",
    description="ML-powered API to detect fake and deceptive reviews",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize database on startup with proper lifespan handling
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown"""
    global _startup_time
    _startup_time = time.time()
    
    logger.info("Starting Fake Review Detection API...")
    
    # Initialize database
    init_db()
    logger.info("Database initialized")
    
    # Initialize NLTK
    init_nltk()
    logger.info("NLTK initialized")
    
    # Load ML models
    try:
        load_models()
        logger.info("ML models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load ML models: {e}")
    
    logger.info("API startup complete")
    
    yield
    
    logger.info("Shutting down Fake Review Detection API...")


# Update app initialization to use lifespan
app = FastAPI(
    title="Fake Review Detection API",
    description="ML-powered API to detect fake and deceptive reviews",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware with security improvements
# Get allowed origins from config or use default restricted list
ALLOWED_ORIGINS = getattr(config, 'ALLOWED_ORIGINS', ["http://localhost:3000", "http://localhost:8080"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Restricted instead of "*"
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)

# Add rate limiting middleware
app.middleware("http")(rate_limit_middleware)


# ==================== HEALTH CHECK ====================

@app.get("/api/v1/health")
async def health_check():
    """Enhanced health check with model status"""
    model_loaded = is_model_loaded()
    uptime_seconds = time.time() - _startup_time if _startup_time else 0
    
    return {
        "status": "OK" if model_loaded else "DEGRADED",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Fake Review Detection API",
        "model_loaded": model_loaded,
        "uptime_seconds": round(uptime_seconds, 2)
    }


# ==================== PREDICTION ENDPOINTS ====================

def validate_review_input(review: str) -> tuple[bool, str]:
    """Validate review input"""
    if not review or not isinstance(review, str):
        return False, "Review must be a non-empty string"
    words = review.split()
    if len(words) < 4:
        return False, "Review must contain at least 4 words"
    if len(words) > 5000:
        return False, "Review cannot exceed 5000 words"
    return True, ""


def validate_rating_input(rating: int) -> tuple[bool, str]:
    """Validate rating input"""
    if not isinstance(rating, int) or rating < 1 or rating > 5:
        return False, "Rating must be an integer between 1 and 5"
    return True, ""


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_review(request: PredictionRequest, request_obj: Request, db: Session = Depends(get_db)):
    """
    Analyze a single review for fake probability
    """
    request_id = request_obj.headers.get("X-Request-ID", "unknown")
    logger.info(f"Prediction request [{request_id}]: rating={request.rating}, text_length={len(request.review)}")
    
    try:
        # Validate inputs
        is_valid, error = validate_review_input(request.review)
        if not is_valid:
            logger.warning(f"Validation failed [{request_id}]: {error}")
            raise HTTPException(status_code=400, detail=error)

        is_valid, error = validate_rating_input(request.rating)
        if not is_valid:
            logger.warning(f"Rating validation failed [{request_id}]: {error}")
            raise HTTPException(status_code=400, detail=error)

        # Get ML prediction
        prediction_result = ml_predict(request.review, request.rating)
        
        logger.info(f"Prediction result [{request_id}]: verdict={prediction_result['verdict']}, fake_score={prediction_result['fake_score']:.4f}")

        # Store in database
        db_prediction = Prediction(
            review_text=request.review,
            rating=request.rating,
            fake_score=prediction_result["fake_score"],
            trust_score=prediction_result["trust_score"],
            verdict=prediction_result["verdict"],
            confidence=prediction_result["confidence"],
            review_length=prediction_result["review_length"],
            word_count=prediction_result["word_count"],
            sentiment=prediction_result["sentiment"],
            exclamation_count=prediction_result["exclamation_count"],
            question_count=prediction_result["question_count"],
            caps_ratio=prediction_result["caps_ratio"],
            repetition=prediction_result["repetition"],
            avg_word_length=prediction_result["avg_word_length"],
        )
        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)

        return PredictionResponse(
            id=db_prediction.id,
            review_text=request.review,
            rating=request.rating,
            **prediction_result,
            created_at=db_prediction.created_at
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error [{request_id}]: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/api/v1/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest, request_obj: Request, db: Session = Depends(get_db)):
    """
    Analyze multiple reviews in batch (max 100)
    Optimized with bulk insert for better performance
    """
    request_id = request_obj.headers.get("X-Request-ID", "unknown")
    logger.info(f"Batch prediction request [{request_id}]: {len(request.reviews)} reviews")
    
    try:
        start_time = time.time()
        results = []
        processed = 0
        predictions_to_insert = []

        # First pass: validate and get predictions
        for item in request.reviews:
            try:
                is_valid, error = validate_review_input(item.review)
                if not is_valid:
                    logger.warning(f"Review validation failed: {error}")
                    continue
                is_valid, error = validate_rating_input(item.rating)
                if not is_valid:
                    logger.warning(f"Rating validation failed: {error}")
                    continue

                prediction_result = ml_predict(item.review, item.rating)

                # Prepare bulk insert data
                predictions_to_insert.append({
                    "review_text": item.review,
                    "rating": item.rating,
                    "fake_score": prediction_result["fake_score"],
                    "trust_score": prediction_result["trust_score"],
                    "verdict": prediction_result["verdict"],
                    "confidence": prediction_result["confidence"],
                    "review_length": prediction_result["review_length"],
                    "word_count": prediction_result["word_count"],
                    "sentiment": prediction_result["sentiment"],
                    "exclamation_count": prediction_result["exclamation_count"],
                    "question_count": prediction_result["question_count"],
                    "caps_ratio": prediction_result["caps_ratio"],
                    "repetition": prediction_result["repetition"],
                    "avg_word_length": prediction_result["avg_word_length"],
                })
                processed += 1
            except Exception as e:
                logger.error(f"Error processing item: {str(e)}")
                continue

        # Bulk insert all predictions at once (major performance improvement)
        if predictions_to_insert:
            db.bulk_insert_mappings(Prediction, predictions_to_insert)
            db.commit()
            
            # Fetch back the inserted records to get IDs and timestamps
            # This is more efficient than individual commits
            inserted_ids = db.query(Prediction.id).order_by(
                Prediction.id.desc()
            ).limit(processed).all()
            
            # Reverse to match insertion order
            inserted_ids = [row[0] for row in inserted_ids][::-1]
            
            # Build response
            for i, pred_data in enumerate(predictions_to_insert):
                if i < len(inserted_ids):
                    results.append(PredictionResponse(
                        id=inserted_ids[i],
                        review_text=pred_data["review_text"],
                        rating=pred_data["rating"],
                        fake_score=pred_data["fake_score"],
                        trust_score=pred_data["trust_score"],
                        verdict=pred_data["verdict"],
                        confidence=pred_data["confidence"],
                        review_length=pred_data["review_length"],
                        word_count=pred_data["word_count"],
                        sentiment=pred_data["sentiment"],
                        exclamation_count=pred_data["exclamation_count"],
                        question_count=pred_data["question_count"],
                        caps_ratio=pred_data["caps_ratio"],
                        created_at=datetime.utcnow()
                    ))

        processing_time = (time.time() - start_time) * 1000
        logger.info(f"Batch prediction completed [{request_id}]: {processed}/{len(request.reviews)} processed in {processing_time:.2f}ms")

        return BatchPredictionResponse(
            total=len(request.reviews),
            processed=processed,
            results=results,
            processing_time_ms=round(processing_time, 2)
        )

    except Exception as e:
        logger.error(f"Batch prediction error [{request_id}]: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


# ==================== ANALYTICS ENDPOINTS ====================

@app.get("/api/v1/predictions", response_model=PredictionHistoryResponse)
async def get_predictions(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    verdict: str = Query(None),
    db: Session = Depends(get_db)
):
    """Get prediction history with pagination"""
    try:
        query = db.query(Prediction)
        if verdict:
            query = query.filter(Prediction.verdict == verdict.upper())

        total = query.count()
        offset = (page - 1) * page_size
        results = query.order_by(desc(Prediction.created_at)).offset(offset).limit(page_size).all()

        return PredictionHistoryResponse(
            total=total,
            page=page,
            page_size=page_size,
            results=[PredictionHistoryItem.from_orm(r) for r in results]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching predictions: {str(e)}")


@app.get("/api/v1/predictions/{prediction_id}", response_model=PredictionResponse)
async def get_prediction_by_id(prediction_id: int, db: Session = Depends(get_db)):
    """Get detailed information about a specific prediction"""
    try:
        prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction not found")
        return PredictionResponse.from_orm(prediction)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching prediction: {str(e)}")


@app.get("/api/v1/analytics/stats", response_model=PredictionStats)
async def get_statistics(db: Session = Depends(get_db)):
    """Get overall statistics about predictions"""
    try:
        total = db.query(func.count(Prediction.id)).scalar() or 0
        fake_count = db.query(func.count(Prediction.id)).filter(Prediction.verdict == "FAKE").scalar() or 0
        genuine_count = db.query(func.count(Prediction.id)).filter(Prediction.verdict == "GENUINE").scalar() or 0
        suspicious_count = db.query(func.count(Prediction.id)).filter(Prediction.verdict == "SUSPICIOUS").scalar() or 0
        avg_confidence = db.query(func.avg(Prediction.confidence)).scalar() or 0
        fake_percentage = (fake_count / total * 100) if total > 0 else 0
        most_recent = db.query(Prediction.created_at).order_by(desc(Prediction.created_at)).first()

        return PredictionStats(
            total_predictions=total,
            fake_detected=fake_count,
            genuine_detected=genuine_count,
            suspicious_detected=suspicious_count,
            fake_percentage=round(fake_percentage, 2),
            average_confidence=round(avg_confidence, 2),
            most_recent=most_recent[0] if most_recent else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching statistics: {str(e)}")


@app.get("/api/v1/model/info", response_model=ModelInfoResponse)
async def get_model_info(db: Session = Depends(get_db)):
    """Get information about the ML model"""
    return ModelInfoResponse(
        version="1.0.0",
        accuracy=0.90,
        precision=0.92,
        recall=0.85,
        f1_score=0.88,
        trained_date=datetime(2025, 3, 5),
        total_features=15011,
        threshold=config.FAKE_THRESHOLD,
        last_updated=datetime.utcnow()
    )


# ==================== APPEAL ENDPOINTS ====================

@app.post("/api/v1/predictions/{prediction_id}/appeal", response_model=AppealResponse)
async def appeal_prediction(
    prediction_id: int,
    request: AppealRequest,
    db: Session = Depends(get_db)
):
    """Submit an appeal for a prediction verdict"""
    try:
        prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction not found")
        if prediction.appeal_submitted:
            raise HTTPException(status_code=400, detail="Appeal already submitted")

        prediction.appeal_submitted = True
        prediction.appeal_reason = request.reason
        db.commit()

        return AppealResponse(
            prediction_id=prediction_id,
            status="submitted",
            message=f"Appeal submitted successfully. ID: {prediction_id}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting appeal: {str(e)}")


# ==================== ROOT ENDPOINT ====================

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Fake Review Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "main_endpoint": "/api/v1/predict"
    }
