# ✈️ Flight Delay Prediction

> **⚠️ PROJECT UNDER CONSTRUCTION** - Week 2 in progress

A machine learning project to predict flight arrival delays using historical US domestic flight data from 2015.

---

## 📊 Project Overview

**Objective:** Predict flight arrival delays (in minutes) using only pre-flight information available before departure.

**Dataset:** 
- Source: US Department of Transportation (2015)
- Size: 4.5M flights after cleaning
- Features: 25+ engineered features
- Target: `ARRIVAL_DELAY` (continuous, regression task)

**Business Use Case:** Enable passengers and airlines to make informed decisions about travel plans and resource allocation based on predicted delays.

---

## 🎯 Project Status

### ✅ Completed (Week 1-2)

- [x] **Exploratory Data Analysis (EDA)**
  - Data cleaning (removed cancelled/diverted flights)
  - Outlier treatment (winsorization at 99th percentile)
  - Feature selection (removed data leakage risks)
  - Correlation analysis
  - Initial feature engineering

- [x] **Data Engineering Pipeline**
  - Automated data splitting (60% train / 20% validation / 20% test)
  - Sklearn-style feature engineering class
  - Production-ready pipeline script
  - Feature engineer can be saved/loaded for deployment

- [x] **Feature Engineering**
  - Target encoding for airports (ORIGIN_AVG_DELAY)
  - One-hot encoding for top 10 busiest airports
  - Traffic volume metrics
  - Airport type classification (major vs regional)

### 🚧 In Progress (Week 2)

- [ ] Data validation pipeline (Great Expectations)
- [ ] Baseline model training (Linear Regression, Random Forest)
- [ ] Model evaluation and comparison
- [ ] Hyperparameter tuning

### 📋 Planned (Week 3+)

- [ ] Advanced models (XGBoost, LightGBM)
- [ ] Feature importance analysis
- [ ] Model interpretability (SHAP values)
- [ ] Final model selection and deployment preparation

---

## 🗂️ Project Structure

```
flight-delay-prediction/
│
├── data/
│   ├── raw/                    # Original dataset (gitignored)
│   ├── interim/                # Cleaned dataset (gitignored)
│   └── processed/              # Train/val/test splits with features (gitignored)
│
├── notebooks/
│   └── 01_initial_eda.ipynb   # Exploratory data analysis
│
├── src/
│   ├── data/
│   │   ├── make_dataset.py    # Data splitting functions
│   │   └── prepare_data.py    # Main pipeline script
│   │
│   └── features/
│       └── build_features.py  # Feature engineering class
│
├── models/                     # Saved models and transformers (gitignored)
│
├── docs/
│   └── week1_eda_report.md    # Detailed EDA documentation
│
└── README.md                   # This file
```

---

## 🚀 How to Use

### Prerequisites

```bash
python 3.8+
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

### Setup

```bash
# Clone the repository
git clone https://github.com/glisicstefan/flight-delay-prediction.git
cd flight-delay-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt  # (to be created)
```

### Running the Pipeline

**Prepare data (split + feature engineering):**
```bash
python src/data/prepare_data.py
```

This will:
1. Load cleaned dataset from `data/interim/flights_cleaned.csv`
2. Create 60/20/20 train/validation/test splits
3. Fit feature engineer on training data
4. Transform all datasets with engineered features
5. Save processed datasets to `data/processed/`
6. Save fitted feature engineer to `models/feature_engineer.pkl`

---

## 📈 Key Features Engineered

| Feature | Type | Description |
|---------|------|-------------|
| `ORIGIN_AVG_DELAY` | Target Encoding | Historical average delay per origin airport |
| `A_{AIRPORT}` | One-Hot | Binary indicators for top 10 busiest airports |
| `IS_MAJOR` | Binary | Major airport (IATA code) vs regional (numeric ID) |
| `ORIGIN_TRAFFIC` | Numeric | Flight volume per airport (proxy for congestion) |

**Data Leakage Prevention:** All statistics computed from training data only and applied to validation/test sets.

---

## 📊 Dataset Statistics

| Split | Rows | Percentage |
|-------|------|------------|
| Train | 2,742,723 | 60% |
| Validation | 914,241 | 20% |
| Test | 914,242 | 20% |

**Features:** 25 original + 14 engineered = **39 total features**

---

## 🔍 Data Leakage Safeguards

**Removed Features (Post-Flight Information):**
- Actual departure/arrival times
- Taxi times, air time, elapsed time
- Delay breakdown categories (airline delay, weather delay, etc.)
- Aircraft identifiers

**Retained Features (Pre-Flight Information):**
- Scheduled departure/arrival times
- Airline, origin, destination
- Temporal features (month, day, day of week)
- Distance, scheduled flight duration

---

## 📝 Documentation

- **Notebooks:** Interactive analysis in `notebooks/`
- **Code Documentation:** Docstrings in all Python modules

---

## 🎓 Learning Goals

This project is designed to demonstrate:

✅ **Data Engineering Best Practices**
- Preventing data leakage
- Proper train/validation/test splitting
- Production-ready feature engineering pipelines

✅ **Machine Learning Workflow**
- End-to-end ML pipeline from raw data to predictions
- Model evaluation and comparison
- Hyperparameter tuning

✅ **Code Quality**
- Modular, reusable code structure
- Sklearn-style transformers
- Version control with Git

---

## 📄 License

This project is for educational and portfolio purposes.

---

## 🙏 Acknowledgments

- Dataset: US Department of Transportation
- Inspiration: Real-world flight delay prediction systems

---

**Last Updated:** February 2026  
**Status:** Week 2 - Data Engineering ✅ | Model Training 🚧
