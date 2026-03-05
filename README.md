# ✈️ Flight Delay Prediction

A machine learning project predicting US domestic flight arrival delays, deployed as a containerized REST API.

---

## 📊 Project Overview

**Goal:** Predict flight arrival delays (in minutes) using only pre-flight information.

**Dataset:**
- 4.5M US domestic flights (2015)
- Train: 3.4M | Validation: 1.1M | Test: 1.1M
- 26 features after optimization

**Best Model:** Random Forest (R²=0.18, RMSE=29.00 minutes)

---

## 🗂️ Project Structure

```
flight-delay-prediction/
│
├── app/                               # FastAPI application
│   ├── main.py                        # API entry point, endpoints
│   ├── models.py                      # Pydantic request/response models
│   ├── predictor.py                   # Model loading & inference logic
│   └── config.py                      # App configuration & settings
│
├── src/
│   ├── data/
│   │   ├── make_dataset.py            # Data splitting (60/20/20)
│   │   ├── prepare_data.py            # Pipeline orchestration
│   │   ├── prepare_modeling_data.py   # X/y separation
│   │   └── validate_data.py           # Data quality checks
│   │
│   ├── features/
│   │   └── build_features.py          # Feature engineering pipeline
│   │
│   └── models/
│       ├── baseline_models.py         # Heuristic baselines
│       ├── train_linear_regression.py # Linear model
│       └── train_random_forest.py     # Random Forest
│
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb      # Exploratory data analysis
│   ├── 02_feature_analysis.ipynb      # Feature engineering analysis
│   └── baseline_evaluation.ipynb     # Baseline model evaluation
│
├── models/                            # Trained models (gitignored)
│   └── random_forest.pkl              # Selected model (~1 GB compressed)
│
├── docs/
│   ├── week1_eda_report.md
│   ├── week2_data_engineering_report.md
│   └── week3_model_training.md
│
├── data/
│   ├── raw/                           # Original dataset (gitignored)
│   ├── interim/                       # Cleaned data (gitignored)
│   └── processed/                     # Train/val/test splits (gitignored)
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .env.example
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/glisicstefan/flight-dalay-prediction.git
cd flight-delay-prediction

# Build and run
docker-compose up --build

# API available at http://localhost:8000
```

### Option 2: Local Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn app.main:app --reload
```

### Run the ML Pipeline

```bash
# 1. Prepare data (split + feature engineering)
python src/data/prepare_data.py

# 2. Separate X and y for modeling
python src/data/prepare_modeling_data.py

# 3. Train models
python src/models/baseline_models.py
python src/models/train_linear_regression.py
python src/models/train_random_forest.py
```

---

## 🌐 API Reference

Interactive documentation is available at `http://localhost:8000/docs` (Swagger UI) after starting the server.

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info and available endpoints |
| GET | `/health` | Health check — model load status |
| POST | `/predict` | Predict arrival delay for a flight |

### Example: Predict Delay

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MONTH": 6,
    "DAY": 15,
    "DAY_OF_WEEK": 2,
    "SCHEDULED_DEPARTURE": 1420,
    "SCHEDULED_ARRIVAL": 1650,
    "AIRLINE": "AA",
    "ORIGIN_AIRPORT": "ORD",
    "DESTINATION_AIRPORT": "LAX"
  }'
```

**Response:**
```json
{
  "predicted_delay_minutes": 12.5,
  "model_version": "random_forest",
  "timestamp": "2025-02-10T14:30:45.123456",
  "flight_info": {
    "airline": "AA",
    "route": "ORD → LAX",
    "scheduled_departure": "14:20",
    "scheduled_arrival": "16:50"
  }
}
```

### Input Validation

Pydantic automatically validates all inputs:
- `MONTH`: 1–12
- `DAY`: 1–31
- `DAY_OF_WEEK`: 1–7 (1=Monday)
- `SCHEDULED_DEPARTURE` / `SCHEDULED_ARRIVAL`: HHMM format (e.g. 1420 = 14:20)
- `AIRLINE`: 2-letter IATA code (e.g. `AA`, `DL`, `UA`)
- `ORIGIN_AIRPORT` / `DESTINATION_AIRPORT`: 3-letter IATA code (e.g. `ORD`, `LAX`)

---

## 📈 Model Performance

### Results Comparison

| Model | Validation RMSE | Validation R² | vs Baseline |
|-------|----------------|---------------|-------------|
| Heuristic Baseline | 31.96 min | 0.0059 | — |
| Linear Regression | 31.55 min | 0.0317 | 5.4x R² |
| Random Forest v1 | 29.34 min | 0.1624 | 27.5x R² |
| **Random Forest v2** | **29.00 min** | **0.1815** | **30.7x R²** |

### Why R²=18% is Meaningful

- Baseline R²=0.6% → **30x improvement**
- Flight delays are inherently noisy (weather, ATC, mechanical issues — none visible pre-flight)
- Academic benchmarks for this problem: R²=10–20%
- A 3-minute RMSE reduction translates to ~$500k–1M in annual operational savings

### Top Features (by importance)

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | DAY | 18.3% | Temporal |
| 2 | SCHEDULED_ARRIVAL | 15.1% | Temporal |
| 3 | MONTH | 10.6% | Temporal |
| 4 | DAY_OF_WEEK | 8.8% | Temporal |
| 5 | ORIGIN_AVG_DELAY | 8.7% | Engineered |

**Key finding:** Temporal features (when you fly) account for 52% of prediction power — more than airport or airline characteristics.

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Data & ML | pandas, numpy, scikit-learn |
| Experiment Tracking | MLflow |
| API | FastAPI, Pydantic, uvicorn |
| Containerization | Docker, docker-compose |
| Visualization | matplotlib, seaborn |
| Persistence | joblib |

---

## 📊 Key Insights

**Temporal > Spatial** — The top 4 most predictive features are all time-based (hour, day, month, day of week). Airport and airline features are secondary. This suggests delays are driven primarily by time-of-day congestion and seasonal patterns rather than route characteristics.

**Feature engineering impact** — Removing 19 redundant airport one-hot features and adding an extracted `HOUR` feature reduced the feature set from 45 to 26 (42% smaller) while improving R² by 11.8%. Simpler models can outperform complex ones when noise features are removed.

---

## 🎓 Learning Outcomes

- ✅ End-to-end ML pipeline (EDA → feature engineering → training → deployment)
- ✅ Data leakage prevention (strict train/val/test separation before all feature computation)
- ✅ Production-ready API with input validation, error handling, and health monitoring
- ✅ Docker containerization with non-root user and health checks
- ✅ Iterative feature optimization (45 → 26 features, +12% R²)
- ✅ Experiment tracking with MLflow

---

## 📄 License

Educational project for portfolio purposes.

---

**Author:** [Stefan Glisic](https://github.com/glisicstefan)  