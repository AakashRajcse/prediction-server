"""
Configuration settings for Fake Review Detection system
Supports environment variables and .env file configuration
"""
import os
from typing import List

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Model file paths
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
TFIDF_PATH = os.path.join(MODELS_DIR, "tfidf.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# Training configuration
TRAINING_DATA_PATH = os.path.join(PROJECT_ROOT, "Sample-Data", "synthetic_fake_reviews_30mb.csv")
SAMPLE_SIZE = 20000
TEST_SIZE = 0.2
CV_FOLDS = 5
RANDOM_STATE = 42

# TF-IDF configuration
MAX_FEATURES = 15000
NGRAM_RANGE = (1, 2)
MIN_DF = 3
MAX_DF = 0.9

# Model configuration
MAX_ITER = 1000
CLASS_WEIGHT = "balanced"

# Prediction configuration - can be overridden by environment variable
FAKE_THRESHOLD = float(os.getenv("FAKE_THRESHOLD", "0.65"))

# Feature columns
NUMERIC_FEATURES = [
    "rating",
    "review_length",
    "word_count",
    "sentiment",
    "exclamation_count",
    "question_count",
    "caps_ratio",
    "repetition",
    "avg_word_length",
    "rating_sentiment_gap",
    "sentiment_consistency"
]

ALL_FEATURES = ["clean_review"] + NUMERIC_FEATURES

# ==================== SECURITY & PERFORMANCE CONFIG ====================

# API Security
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:8080"
).split(",")

# Rate limiting
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))  # requests per minute
RATE_LIMIT_BURST = int(os.getenv("RATE_LIMIT_BURST", "10"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "")

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fake_review_detection.db")

# API
API_VERSION = "1.0.0"
API_TITLE = "Fake Review Detection API"
