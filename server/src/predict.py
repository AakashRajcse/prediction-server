"""
Prediction module for Fake Review Detection
Reusable functions for both CLI and API
"""
import pickle
import numpy as np
import os
import sys

# Import local modules
from .utils import clean_text, extract_numeric_features, init_nltk
from . import config

# Global model components (lazy loaded)
_model = None
_tfidf = None
_scaler = None
_models_loaded = False


def load_models():
    """Load pre-trained model components"""
    global _model, _tfidf, _scaler, _models_loaded
    
    if _models_loaded:
        return

    # Check if model files exist
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {config.MODEL_PATH}")
    if not os.path.exists(config.TFIDF_PATH):
        raise FileNotFoundError(f"TF-IDF not found: {config.TFIDF_PATH}")
    if not os.path.exists(config.SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found: {config.SCALER_PATH}")

    # Load with proper file handling
    with open(config.MODEL_PATH, "rb") as f:
        _model = pickle.load(f)
    with open(config.TFIDF_PATH, "rb") as f:
        _tfidf = pickle.load(f)
    with open(config.SCALER_PATH, "rb") as f:
        _scaler = pickle.load(f)

    _models_loaded = True


def is_model_loaded():
    """Check if models are loaded"""
    return _models_loaded


def create_features(review, rating):
    """
    Create feature vector for prediction

    Args:
        review: Review text
        rating: Review rating (1-5)

    Returns:
        np.ndarray: Feature vector for model prediction
    """
    # Ensure models are loaded
    if not _models_loaded:
        load_models()

    # Validate inputs
    if not isinstance(review, str) or not review.strip():
        raise ValueError("Review must be a non-empty string")

    if not isinstance(rating, (int, float)) or rating < 1 or rating > 5:
        raise ValueError("Rating must be a number between 1 and 5")

    # Clean text
    clean = clean_text(review)

    # Get TF-IDF features
    text_vector = _tfidf.transform([clean]).toarray()

    # Extract numeric features
    numeric_dict = extract_numeric_features(review, rating)

    # Create numeric array in correct column order
    numeric_array = np.array([[
        numeric_dict["rating"],
        numeric_dict["review_length"],
        numeric_dict["word_count"],
        numeric_dict["sentiment"],
        numeric_dict["exclamation_count"],
        numeric_dict["question_count"],
        numeric_dict["caps_ratio"],
        numeric_dict["repetition"],
        numeric_dict["avg_word_length"],
        numeric_dict["rating_sentiment_gap"],
        numeric_dict["sentiment_consistency"]
    ]])

    # Scale numeric features
    numeric_scaled = _scaler.transform(numeric_array)

    # Combine all features
    final_features = np.hstack((text_vector, numeric_scaled))

    return final_features


def predict(review, rating):
    """
    Predict if a review is fake or genuine
    
    Returns dict with prediction results (for API use)

    Args:
        review: Review text
        rating: Review rating (1-5)
        
    Returns:
        dict: Prediction results with fake_score, trust_score, verdict, confidence, and features
    """
    # Ensure models are loaded
    if not _models_loaded:
        load_models()

    # Create features
    features = create_features(review, rating)

    # Make prediction
    prob = _model.predict_proba(features)[0][1]

    # Determine verdict based on threshold
    threshold = config.FAKE_THRESHOLD
    if prob > threshold:
        verdict = "FAKE"
    elif prob > (threshold - 0.15):
        verdict = "SUSPICIOUS"
    else:
        verdict = "GENUINE"

    # Calculate confidence
    confidence = max(prob, 1 - prob) * 100

    # Get numeric features for response
    numeric_dict = extract_numeric_features(review, rating)

    return {
        "fake_score": round(prob, 4),
        "trust_score": round(1 - prob, 4),
        "verdict": verdict,
        "confidence": round(confidence, 2),
        "review_length": numeric_dict.get("review_length"),
        "word_count": numeric_dict.get("word_count"),
        "sentiment": round(numeric_dict.get("sentiment", 0), 4),
        "exclamation_count": numeric_dict.get("exclamation_count"),
        "question_count": numeric_dict.get("question_count"),
        "caps_ratio": round(numeric_dict.get("caps_ratio", 0), 4),
        "repetition": numeric_dict.get("repetition"),
        "avg_word_length": round(numeric_dict.get("avg_word_length", 0), 4),
    }


def predict_cli(review, rating):
    """
    Predict and display results (for CLI use)

    Args:
        review: Review text
        rating: Review rating (1-5)
    """
    result = predict(review, rating)

    # Display results
    print("\n" + "=" * 40)
    print("Analysis Result")
    print("=" * 40)
    print(f"Fake Score:   {result['fake_score']:.2%}")
    print(f"Trust Score:  {result['trust_score']:.2%}")

    if result["verdict"] == "FAKE":
        print("\n[ALERT] Fake Review Detected")
    else:
        print("\n[OK] Genuine Review")
    print("=" * 40)


def main():
    """Interactive prediction loop"""
    print("\n" + "=" * 50)
    print("   FAKE REVIEW DETECTION SYSTEM")
    print("=" * 50)

    # Load models once
    try:
        load_models()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("Please train the model first: python src/train.py")
        return

    # Initialize NLTK
    init_nltk()

    print("\nEnter review information (type 'quit' to exit):\n")

    while True:
        print("-" * 40)

        try:
            rating_input = input("Enter Rating (1-5): ").strip()

            if rating_input.lower() == "quit":
                print("\n👋 Program Ended")
                break

            rating = float(rating_input)

            if rating < 1 or rating > 5:
                print("[WARN] Rating must be between 1 and 5")
                continue

        except ValueError:
            print("[WARN] Please enter a valid number for rating")
            continue

        review = input("Enter Review: ").strip()

        if not review:
            print("[WARN] Review cannot be empty")
            continue

        if len(review.split()) < 4:
            print("[WARN] Review too short (minimum 4 words for reliable analysis)")
            continue

        predict_cli(review, rating)

        cont = input("\nCheck another review? (y/n): ").strip().lower()
        if cont != "y":
            print("\nProgram Ended")
            break


if __name__ == "__main__":
    main()
