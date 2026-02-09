"""
Configuration module for Flight Delay Prediction API.

Centralizes all paths, constants, and settings in one place.
Following 12-Factor App principles: https://12factor.net/config
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings


# ============================================================================
# 1. PROJECT PATHS
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent  # flight-delay-prediction/

DATA_DIR = BASE_DIR / "data"          # Raw/processed data (optional for API)
MODELS_DIR = BASE_DIR / "models"      # Trained models
LOGS_DIR = BASE_DIR / "logs"          # Application logs

# Ensure directories exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Model file paths
MODEL_PATH = MODELS_DIR / "random_forest.pkl"
FEATURE_ENGINEER_PATH = MODELS_DIR / "feature_engineer.pkl"
DATA_VALIDATOR_PATH = MODELS_DIR / "data_validator.pkl"


# ============================================================================
# 2. MODEL METADATA
# ============================================================================

N_FEATURES = 25

FEATURE_NAMES: List[str] = [
    # Temporal features
    'MONTH', 
    'DAY', 
    'DAY_OF_WEEK', 
    'HOUR',
    
    # Time features
    'SCHEDULED_ARRIVAL', 
    
    # Delay features
    'ORIGIN_AVG_DELAY', 
    'DESTINATION_AVG_DELAY',

    # Airlines (one-hot encoded)
    'AIRLINE_WN', 'AIRLINE_DL', 'AIRLINE_AA', 'AIRLINE_OO', 
    'AIRLINE_EV', 'AIRLINE_UA', 'AIRLINE_MQ', 'AIRLINE_B6', 
    'AIRLINE_US', 'AIRLINE_AS', 'AIRLINE_NK', 'AIRLINE_F9', 
    'AIRLINE_HA', 'AIRLINE_VX',
    
    # Airport traffic features
    'ORIGIN_TRAFFIC', 
    'DESTINATION_TRAFFIC',
    'IS_MAJOR_ORIGIN', 
    'IS_MAJOR_DESTINATION'
]

MODEL_VERSION = "random_forest"  

# Target variable info
TARGET_NAME = "ARRIVAL_DELAY"
TARGET_UNIT = "minutes"

# ============================================================================
# 3. VALIDATION RANGES (Business Logic)
# ============================================================================

VALIDATION_RANGES = {
    # Temporal features
    "MONTH": {"min": 1, "max": 12},
    "DAY": {"min": 1, "max": 31},
    "DAY_OF_WEEK": {"min": 1, "max": 7},
    "HOUR": {"min": 0, "max": 23},
    
    # Time features (in HHMM format: 0000-2359)
    "SCHEDULED_DEPARTURE": {"min": 0, "max": 2359},
    "SCHEDULED_ARRIVAL": {"min": 0, "max": 2359},
    
    # Distance (miles)
    "DISTANCE": {"min": 0, "max": 5000},  # US domestic max ~3000, buffer to 5000
    
    # Delay features (minutes, can be negative = early)
    "ORIGIN_AVG_DELAY": {"min": -60, "max": 300},  # -60min early to 5hr late
    "DESTINATION_AVG_DELAY": {"min": -60, "max": 300},
    
    # Traffic features (flight counts)
    "ORIGIN_TRAFFIC": {"min": 0, "max": 100000},
    "DESTINATION_TRAFFIC": {"min": 0, "max": 100000},
    
    # Binary features
    "IS_MAJOR_ORIGIN": {"min": 0, "max": 1},
    "IS_MAJOR_DESTINATION": {"min": 0, "max": 1},
}

# Valid airline codes (optional - for stricter validation)
# Based on US domestic carriers
VALID_AIRLINES: Optional[List[str]] = [
    'WN', 'DL', 'AA', 'OO', 'EV', 'UA', 'MQ', 'B6', 
    'US', 'AS', 'NK', 'F9', 'HA', 'VX'
]

# Valid airports - too many to list, will use validator fallback
VALID_AIRPORTS: Optional[List[str]] = None  # Use data_validator.pkl instead


# ============================================================================
# 4. API SETTINGS (Environment Variables)
# ============================================================================

class Settings(BaseSettings):
    """
    API configuration using Pydantic Settings.
    
    Reads from environment variables (.env file) with fallback defaults.
    
    Example .env file:
        API_HOST=0.0.0.0
        API_PORT=8000
        LOG_LEVEL=INFO
        DEBUG=False
    """
    
    # API server settings
    API_HOST: str = "127.0.0.1"      # localhost for dev, 0.0.0.0 for Docker
    API_PORT: int = 8000             # Standard FastAPI port
    API_VERSION: str = "v1"          # API version prefix
    
    # Logging settings
    LOG_LEVEL: str = "INFO"          # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FILE: str = "api.log"        # Log filename
    
    # Debug mode
    DEBUG: bool = True               # True for dev (verbose errors), False for prod
    
    # Model loading
    LOAD_MODEL_ON_STARTUP: bool = True  # Load model when API starts
    
    # CORS settings (if needed for frontend)
    ALLOW_CORS: bool = True
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Allow extra fields (won't error on unknown env vars)
        extra = "ignore"


# Instantiate settings (reads from .env if exists)
settings = Settings()

# ============================================================================
# 5. LOGGING CONFIGURATION
# ============================================================================

LOG_FILE_PATH = LOGS_DIR / settings.LOG_FILE

# Logging format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ============================================================================
# 6. API RESPONSE TEMPLATES
# ============================================================================

# Success response schema
PREDICTION_RESPONSE_EXAMPLE = {
    "predicted_delay_minutes": 12.5,
    "model_version": MODEL_VERSION,
    "timestamp": "2025-02-09T14:23:45Z",
    "confidence_interval": [8.2, 16.8],  # Optional: ±1 std dev
}

# Error response schema
ERROR_RESPONSE_EXAMPLE = {
    "error": "ValidationError",
    "message": "Invalid input data",
    "details": {
        "field": "MONTH",
        "value": 13,
        "expected": "1-12"
    },
    "timestamp": "2025-02-09T14:23:45Z"
}


# ============================================================================
# 7. HEALTH CHECK
# ============================================================================

HEALTH_CHECK_RESPONSE = {
    "status": "healthy",
    "model_loaded": False,  # Will be updated at runtime
    "version": settings.API_VERSION,
    "model_version": MODEL_VERSION,
}


# ============================================================================
# 8. VALIDATION (Helper functions)
# ============================================================================

def validate_feature_ranges(features: dict) -> List[str]:
    """
    Validate feature values against expected ranges.
    
    Args:
        features: Dictionary of feature name -> value
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    for feature_name, ranges in VALIDATION_RANGES.items():
        if feature_name in features:
            value = features[feature_name]
            
            # Check numeric range
            if not isinstance(value, (int, float)):
                errors.append(
                    f"{feature_name}: expected numeric value, got {type(value).__name__}"
                )
                continue
            
            # Check min/max bounds
            if value < ranges["min"] or value > ranges["max"]:
                errors.append(
                    f"{feature_name}={value} is out of valid range "
                    f"[{ranges['min']}, {ranges['max']}]"
                )
    
    return errors


def validate_airline(airline_code: str) -> bool:
    """
    Check if airline code is valid.
    
    Args:
        airline_code: Two-letter airline code (e.g., "AA", "DL")
        
    Returns:
        True if valid, False otherwise
    """
    if VALID_AIRLINES is None:
        return True  # Skip validation if list not provided
    
    return airline_code.upper() in VALID_AIRLINES


# ============================================================================
# 9. PRINTING (Debug helpers)
# ============================================================================

def print_config():
    """Print current configuration (for debugging)."""
    print("\n" + "="*70)
    print("FLIGHT DELAY API CONFIGURATION")
    print("="*70)
    
    print("\n📂 Paths:")
    print(f"  BASE_DIR:              {BASE_DIR}")
    print(f"  MODELS_DIR:            {MODELS_DIR}")
    print(f"  MODEL_PATH:            {MODEL_PATH}")
    print(f"  FEATURE_ENGINEER:      {FEATURE_ENGINEER_PATH}")
    print(f"  LOGS_DIR:              {LOGS_DIR}")
    
    print("\n🤖 Model:")
    print(f"  VERSION:               {MODEL_VERSION}")
    print(f"  N_FEATURES:            {N_FEATURES}")
    print(f"  TARGET:                {TARGET_NAME} ({TARGET_UNIT})")
    print(f"  FEATURES:              {', '.join(FEATURE_NAMES[:5])}... (+ {N_FEATURES-5} more)")
    
    print("\n🌐 API Settings:")
    print(f"  HOST:                  {settings.API_HOST}")
    print(f"  PORT:                  {settings.API_PORT}")
    print(f"  VERSION:               {settings.API_VERSION}")
    print(f"  DEBUG:                 {settings.DEBUG}")
    print(f"  LOG_LEVEL:             {settings.LOG_LEVEL}")
    print(f"  LOAD_MODEL_ON_STARTUP: {settings.LOAD_MODEL_ON_STARTUP}")
    
    print("\n✅ Validation:")
    print(f"  Ranges defined:        {len(VALIDATION_RANGES)} features")
    print(f"  Valid airlines:        {len(VALID_AIRLINES) if VALID_AIRLINES else 'Not restricted'}")
    
    print("\n" + "="*70)


def check_model_files() -> dict:
    """
    Check if required model files exist.
    
    Returns:
        Dictionary with file existence status
    """
    status = {
        "model": MODEL_PATH.exists(),
        "feature_engineer": FEATURE_ENGINEER_PATH.exists(),
        "data_validator": DATA_VALIDATOR_PATH.exists(),
    }
    
    print("\n📦 Model Files Status:")
    for name, exists in status.items():
        symbol = "✅" if exists else "❌"
        print(f"  {symbol} {name}: {exists}")
    
    return status


# ============================================================================
# 10. MAIN (Test when run directly)
# ============================================================================

if __name__ == "__main__":
    # Print config
    print_config()
    
    # Check model files
    file_status = check_model_files()
    
    # Test validation
    print("\n🧪 Testing validation...")
    
    # Valid input
    test_features_valid = {
        "MONTH": 6,
        "DAY": 15,
        "HOUR": 14,
        "DISTANCE": 1500,
    }
    errors = validate_feature_ranges(test_features_valid)
    print(f"\n  Valid input: {len(errors)} errors - {errors if errors else '✅ PASS'}")
    
    # Invalid input
    test_features_invalid = {
        "MONTH": 13,  # Out of range!
        "DAY": 15,
        "HOUR": 25,   # Out of range!
        "DISTANCE": -100,  # Negative!
    }
    errors = validate_feature_ranges(test_features_invalid)
    print(f"  Invalid input: {len(errors)} errors")
    for error in errors:
        print(f"    ❌ {error}")
    
    # Test airline validation
    print("\n  Airline validation:")
    print(f"    'AA' valid: {validate_airline('AA')}")
    print(f"    'ZZ' valid: {validate_airline('ZZ')}")