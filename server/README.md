# Server (Backend API)

Complete FastAPI backend for Fake Review Detection System.

## Quick Start

### 1. Install Dependencies
```bash
cd server
pip install -r requirements.txt
```

### 2. Start the Server
```bash
python start.py
```

You should see:
```
[2] Starting API Server on http://localhost:8000
    API Documentation: http://localhost:8000/docs
    Alternative Docs: http://localhost:8000/redoc
```

### 3. Access API
- **Swagger UI (Interactive Docs)**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **API Health**: http://localhost:8000/api/v1/health

---

## Testing with Postman

### Import Collection
1. Open Postman
2. Click `File` → `Import`
3. Select `Postman_Collection.json` from this folder
4. All API endpoints will be ready to test!

### Quick Test Requests

**Health Check:**
```
GET http://localhost:8000/api/v1/health
```

**Single Review Prediction:**
```
POST http://localhost:8000/api/v1/predict
Content-Type: application/json

{
  "review": "This product exceeded my expectations. Excellent quality and fast shipping!",
  "rating": 5
}
```

**Batch Prediction:**
```
POST http://localhost:8000/api/v1/predict/batch
Content-Type: application/json

{
  "reviews": [
    {"review": "Great product!", "rating": 5},
    {"review": "Not good", "rating": 1}
  ]
}
```

**Get Analytics:**
```
GET http://localhost:8000/api/v1/analytics/stats
```

---

## API Endpoints

### 1. Health Check
```
GET /api/v1/health
Response: { status: "OK", timestamp: "...", service: "..." }
```

### 2. Single Review Prediction ⭐
```
POST /api/v1/predict
Body: { review: string, rating: 1-5 }
Response: { id, fake_score, trust_score, verdict, confidence, features... }
```

### 3. Batch Prediction
```
POST /api/v1/predict/batch
Body: { reviews: [{ review, rating }, ...] }
Response: { total, processed, results, processing_time_ms }
```

### 4. Prediction History
```
GET /api/v1/predictions?page=1&page_size=10&verdict=FAKE
Response: { total, page, page_size, results }
```

### 5. Get Single Prediction
```
GET /api/v1/predictions/{id}
Response: { id, review_text, rating, fake_score, verdict, ... }
```

### 6. Analytics Dashboard
```
GET /api/v1/analytics/stats
Response: { total_predictions, fake_detected, genuine_detected, fake_percentage, ... }
```

### 7. Model Information
```
GET /api/v1/model/info
Response: { version, accuracy, precision, recall, f1_score, threshold, ... }
```

### 8. Appeal Prediction
```
POST /api/v1/predictions/{id}/appeal
Body: { reason: string }
Response: { prediction_id, status, message }
```

---

## Project Structure

```
server/
├── api/                    # FastAPI application
│   ├── main.py            # API endpoints
│   ├── database.py        # SQLite setup
│   ├── models.py          # Database models
│   ├── schemas.py         # Request/response validation
│   ├── utils.py           # ML integration
│   └── __init__.py
├── src/                   # ML/Training code
│   ├── config.py          # Configuration
│   ├── utils.py           # ML utilities
│   ├── train.py           # Training script
│   ├── predict.py         # Inference script
│   └── __init__.py
├── models/                # Trained ML models
│   ├── model.pkl          # Logistic Regression
│   ├── tfidf.pkl          # TF-IDF vectorizer
│   └── scaler.pkl         # Feature scaler
├── requirements.txt       # Python dependencies
├── start.py               # Server startup script
├── run_api.py             # Alternative startup
├── test_api_direct.py     # Direct API testing
└── Postman_Collection.json # Postman test collection
```

---

## Important Files

### `start.py` ⭐
Easy startup script. Just run:
```bash
python start.py
```

### `Postman_Collection.json`
Import this into Postman to test all API endpoints without writing code.

### `test_api_direct.py`
Test the API directly without running the server:
```bash
python test_api_direct.py
```

---

## Database

SQLite database is created automatically at runtime:
```
fake_review_detection.db
```

Tables:
- `predictions` - All review predictions
- `model_metadata` - Model information

---

## Configuration

Edit `src/config.py` to change:
- `FAKE_THRESHOLD` - Fake score threshold (default: 0.65)
- `MAX_FEATURES` - TF-IDF vocabulary size (default: 15000)
- `CV_FOLDS` - Cross-validation folds

---

## Troubleshooting

**Port 8000 already in use:**
```bash
# Use different port
python -m uvicorn api.main:app --port 8001
```

**Import errors:**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

**Models not found:**
```bash
# Check models directory
ls models/
```

---

## Next: Frontend

The backend is ready for a React frontend!
- All APIs documented and tested
- CORS enabled for frontend
- Database persistence ready

See `../client/` for frontend setup.

---

**API is ready for production!** 🚀
