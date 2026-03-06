"""
SQLAlchemy database models with optimized indexes
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, Index
from sqlalchemy.sql import func
from .database import Base

class Prediction(Base):
    """Model for storing review predictions"""
    __tablename__ = "predictions"
    
    # Define table-level indexes for query optimization
    __table_args__ = (
        # Index for filtering by verdict (commonly used in analytics)
        Index('ix_predictions_verdict', 'verdict'),
        # Index for sorting by created_at (pagination)
        Index('ix_predictions_created_at', 'created_at'),
        # Composite index for verdict + created_at (common query pattern)
        Index('ix_predictions_verdict_created', 'verdict', 'created_at'),
        # Index for appeals filtering
        Index('ix_predictions_appeal', 'appeal_submitted'),
    )

    id = Column(Integer, primary_key=True, index=True)
    review_text = Column(Text, nullable=False)
    rating = Column(Integer, nullable=False)
    fake_score = Column(Float, nullable=False)
    trust_score = Column(Float, nullable=False)
    verdict = Column(String(20), nullable=False, index=True)  # Add index for verdict
    confidence = Column(Float, nullable=False)

    # Feature values for transparency
    review_length = Column(Integer)
    word_count = Column(Integer)
    sentiment = Column(Float)
    exclamation_count = Column(Integer)
    question_count = Column(Integer)
    caps_ratio = Column(Float)
    repetition = Column(Integer)
    avg_word_length = Column(Float)

    # Metadata
    created_at = Column(DateTime, server_default=func.now(), index=True)  # Add index for sorting
    appeal_submitted = Column(Boolean, default=False, index=True)
    appeal_reason = Column(Text, nullable=True)

    def __repr__(self):
        return f"<Prediction(id={self.id}, verdict={self.verdict}, fake_score={self.fake_score})>"


class ModelMetadata(Base):
    """Model for storing model information"""
    __tablename__ = "model_metadata"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(String(10), nullable=False)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    trained_date = Column(DateTime)
    total_features = Column(Integer)
    threshold = Column(Float, default=0.65)
    last_updated = Column(DateTime, server_default=func.now())

    def __repr__(self):
        return f"<ModelMetadata(version={self.version}, accuracy={self.accuracy})>"
