"""
Training script for Fake Review Detection Model
"""
import pandas as pd
import numpy as np
import pickle
import os
import sys

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack

# Import local modules
from utils import clean_text, extract_numeric_features, init_nltk
import config

def load_data():
    """Load and preprocess the training dataset"""
    print("Loading data...")

    if not os.path.exists(config.TRAINING_DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {config.TRAINING_DATA_PATH}")

    df = pd.read_csv(
        config.TRAINING_DATA_PATH,
        encoding="latin-1",
        on_bad_lines="skip"
    )

    # Rename dataset columns
    df.rename(columns={
        "text_": "reviewText",
        "rating": "overall"
    }, inplace=True)

    # Sample data
    df = df.sample(config.SAMPLE_SIZE, random_state=config.RANDOM_STATE).reset_index(drop=True)

    # Remove missing reviews
    df.dropna(subset=["reviewText"], inplace=True)

    # Convert label
    df["label"] = df["label"].apply(lambda x: 1 if x == "CG" else 0)

    return df

def engineer_features(df):
    """Extract all features from the dataset"""
    print("Engineering features...")

    # Initialize NLTK (only once)
    init_nltk()

    # Text cleaning
    df["clean_review"] = df["reviewText"].apply(clean_text)

    # Extract numeric features
    numeric_features_list = []
    for idx, row in df.iterrows():
        features = extract_numeric_features(row["reviewText"], row["overall"])
        numeric_features_list.append(features)

    numeric_df = pd.DataFrame(numeric_features_list)
    df = pd.concat([df, numeric_df], axis=1)

    print(f"Class Distribution:\n{df['label'].value_counts()}")

    return df

def prepare_features(X, tfidf_vectorizer, scaler, fit=False):
    """
    Prepare features for model training/prediction

    Args:
        X: DataFrame with features
        tfidf_vectorizer: TfidfVectorizer instance
        scaler: StandardScaler instance
        fit: If True, fit the transformers; else just transform
    """
    # Text features via TF-IDF
    if fit:
        X_text = tfidf_vectorizer.fit_transform(X["clean_review"])
    else:
        X_text = tfidf_vectorizer.transform(X["clean_review"])

    # Numeric features
    if fit:
        X_num = scaler.fit_transform(X[config.NUMERIC_FEATURES])
    else:
        X_num = scaler.transform(X[config.NUMERIC_FEATURES])

    # Combine features
    X_final = hstack([X_text, X_num])

    return X_final

def train_model():
    """Main training pipeline"""
    print("Starting Fake Review Detection Model Training...\n")

    # Load data
    df = load_data()

    # Feature engineering
    df = engineer_features(df)

    # Split data
    print(f"Splitting data (test_size={config.TEST_SIZE})...")
    X = df[config.ALL_FEATURES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )

    # Initialize transformers
    tfidf = TfidfVectorizer(
        max_features=config.MAX_FEATURES,
        ngram_range=config.NGRAM_RANGE,
        min_df=config.MIN_DF,
        max_df=config.MAX_DF
    )

    scaler = StandardScaler()

    # Prepare features
    print("Preparing features...")
    X_train_final = prepare_features(X_train, tfidf, scaler, fit=True)
    X_test_final = prepare_features(X_test, tfidf, scaler, fit=False)

    # Train model
    print("Training model...")
    model = LogisticRegression(
        max_iter=config.MAX_ITER,
        class_weight=config.CLASS_WEIGHT
    )
    model.fit(X_train_final, y_train)

    # Cross-validation
    print(f"Cross-validating (cv={config.CV_FOLDS})...")
    cv_scores = cross_val_score(model, X_train_final, y_train, cv=config.CV_FOLDS)
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Evaluation
    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test_final)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save models
    print("\nSaving models...")
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    with open(config.MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(config.TFIDF_PATH, "wb") as f:
        pickle.dump(tfidf, f)

    with open(config.SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"✅ Models saved to {config.MODELS_DIR}/")
    print("=" * 50)
    print("MODEL TRAINING COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    train_model()
