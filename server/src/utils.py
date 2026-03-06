"""
Utility functions for text processing and feature engineering
Optimized with VADER sentiment (faster than TextBlob)
"""
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Try to import VADER, fallback to TextBlob if not available
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _VADER_AVAILABLE = True
except ImportError:
    _VADER_AVAILABLE = False
    from textblob import TextBlob

# Initialize NLTK components once
_NLTK_INITIALIZED = False
_ps = None
_stop_words = None
_vader_analyzer = None

def init_nltk():
    """Download and initialize NLTK resources (run once)"""
    global _NLTK_INITIALIZED, _ps, _stop_words, _vader_analyzer

    if not _NLTK_INITIALIZED:
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("vader_lexicon", quiet=True)
        _ps = PorterStemmer()
        _stop_words = set(stopwords.words("english"))
        
        # Initialize VADER analyzer
        if _VADER_AVAILABLE:
            _vader_analyzer = SentimentIntensityAnalyzer()
        
        _NLTK_INITIALIZED = True

def get_stemmer():
    """Get the PorterStemmer instance"""
    if not _NLTK_INITIALIZED:
        init_nltk()
    return _ps

def get_stopwords():
    """Get the stopwords set"""
    if not _NLTK_INITIALIZED:
        init_nltk()
    return _stop_words

def get_sentiment_analyzer():
    """Get the sentiment analyzer (VADER or TextBlob)"""
    if not _NLTK_INITIALIZED:
        init_nltk()
    return _vader_analyzer

def clean_text(text):
    """
    Clean and preprocess text: lowercase, remove special chars,
    tokenize, remove stopwords, and stem
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in get_stopwords()]
    tokens = [get_stemmer().stem(w) for w in tokens]

    return " ".join(tokens)

def extract_numeric_features(review, rating):
    """
    Extract all numeric features from a review in a single pass
    Optimized with VADER sentiment analysis (faster than TextBlob)

    Returns:
        dict: Feature name to value mapping
    """
    review_str = str(review) if not isinstance(review, str) else review

    # Basic metrics
    review_length = len(review_str)
    words = review_str.split()
    word_count = len(words)

    # Special character counts
    exclamation_count = review_str.count("!")
    question_count = review_str.count("?")

    # Capitalization ratio
    caps_ratio = sum(1 for c in review_str if c.isupper()) / (review_length + 1)

    # Word repetition
    repetition = word_count - len(set(words))

    # Average word length
    avg_word_length = np.mean([len(w) for w in words]) if word_count > 0 else 0

    # Sentiment analysis - use VADER (faster) or fallback to TextBlob
    if _VADER_AVAILABLE and _vader_analyzer is not None:
        # VADER returns 'compound' score between -1 and 1
        sentiment = _vader_analyzer.polarity_scores(review_str)['compound']
    else:
        # Fallback to TextBlob
        sentiment = TextBlob(review_str).sentiment.polarity

    # Derived features
    rating_sentiment_gap = abs(rating - (sentiment * 5))
    sentiment_consistency = sentiment * rating

    return {
        "review_length": review_length,
        "word_count": word_count,
        "sentiment": sentiment,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "caps_ratio": caps_ratio,
        "repetition": repetition,
        "avg_word_length": avg_word_length,
        "rating_sentiment_gap": rating_sentiment_gap,
        "sentiment_consistency": sentiment_consistency,
        "rating": rating
    }
