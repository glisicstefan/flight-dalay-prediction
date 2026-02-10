"""
ML Model Predictor - Handles model loading and inference.

This module encapsulates all ML logic:
- Loading trained models (Random Forest, Feature Engineer)
- Transforming raw input data into model features
- Making predictions
- Handling errors gracefully

Design Pattern: Singleton
- Model is loaded ONCE when API starts
- Shared across all requests (memory efficient)
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging

from app.config import (
    MODEL_PATH,
    FEATURE_ENGINEER_PATH,
    DATA_VALIDATOR_PATH,
    MODEL_VERSION,
    N_FEATURES,
    TARGET_NAME,
    TARGET_UNIT
)
from app.models import FlightData, PredictionResponse, create_flight_info
from datetime import datetime


# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# 1. MODEL LOADER - Singleton Pattern
# ============================================================================

class ModelPredictor:
    """
    ML Model Predictor using Singleton pattern.
    
    Loads model once and reuses it for all predictions.
    Thread-safe and memory efficient.
    
    Usage:
        predictor = ModelPredictor()
        predictor.load_models()
        
        result = predictor.predict(flight_data)
    """
    
    _instance = None  # Singleton instance
    _models_loaded = False
    
    def __new__(cls):
        """
        Singleton pattern - only one instance exists.
        
        This ensures models are loaded only once, not on every request.
        """
        if cls._instance is None:
            logger.info("Creating new ModelPredictor instance")
            cls._instance = super(ModelPredictor, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize predictor (only once due to singleton)."""
        if not ModelPredictor._models_loaded:
            # Initialize attributes
            self.model = None
            self.feature_engineer = None
            self.data_validator = None
            
            # Model metadata
            self.model_version = MODEL_VERSION
            self.n_features = N_FEATURES
            self.target_name = TARGET_NAME
            self.target_unit = TARGET_UNIT
    
    # ========================================================================
    # 2. MODEL LOADING
    # ========================================================================
    
    def load_models(self) -> bool:
        """
        Load all required models from disk.
        
        Returns:
            True if all models loaded successfully, False otherwise
            
        Raises:
            FileNotFoundError: If model files don't exist
            Exception: If model loading fails
        """
        if ModelPredictor._models_loaded:
            logger.info("Models already loaded, skipping...")
            return True
        
        import sys
        from pathlib import Path
        src_path = Path(__file__).parent.parent / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        try:
            logger.info("Loading ML models...")
            
            # Check if files exist
            self._check_model_files()
            
            # Load Random Forest model
            logger.info(f"Loading model from {MODEL_PATH}")
            self.model = joblib.load(MODEL_PATH)
            logger.info(f"✅ Model loaded: {type(self.model).__name__}")
            
            # Load Feature Engineer
            logger.info(f"Loading feature engineer from {FEATURE_ENGINEER_PATH}")
            self.feature_engineer = joblib.load(FEATURE_ENGINEER_PATH)
            logger.info(f"✅ Feature engineer loaded: {type(self.feature_engineer).__name__}")
            
            # Load Data Validator (optional)
            if DATA_VALIDATOR_PATH.exists():
                try:
                    logger.info(f"Loading data validator...")
                    self.data_validator = joblib.load(DATA_VALIDATOR_PATH)
                    logger.info("✅ Data validator loaded")
                except Exception as e:
                    logger.warning(f"⚠️  Data validator loading failed: {e}")
                    logger.warning("   Continuing without validator (optional)")
                    self.data_validator = None
            
            # Verify model
            self._verify_model()
            
            # Mark as loaded
            ModelPredictor._models_loaded = True
            logger.info("🎉 All models loaded successfully!")
            
            return True
            
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _check_model_files(self):
        """
        Check if all required model files exist.
        
        Raises:
            FileNotFoundError: If required files are missing
        """
        required_files = {
            "Model": MODEL_PATH,
            "Feature Engineer": FEATURE_ENGINEER_PATH
        }
        
        missing = []
        for name, path in required_files.items():
            if not path.exists():
                missing.append(f"{name} ({path})")
        
        if missing:
            raise FileNotFoundError(
                f"Required model files not found:\n" + 
                "\n".join(f"  - {m}" for m in missing)
            )
    
    def _verify_model(self):
        """
        Verify loaded model works correctly.
        
        Makes a test prediction with dummy data.
        
        Raises:
            Exception: If model verification fails
        """
        logger.info("Verifying model with test prediction...")
        
        try:
            # Create dummy input (one flight)
            dummy_data = pd.DataFrame({
                'MONTH': [6],
                'DAY': [15],
                'DAY_OF_WEEK': [2],
                'SCHEDULED_DEPARTURE': [1420],
                'SCHEDULED_ARRIVAL': [1650],
                'AIRLINE': ['AA'],
                'ORIGIN_AIRPORT': ['ORD'],
                'DESTINATION_AIRPORT': ['LAX']
            })
            
            # Transform features
            features = self.feature_engineer.transform(dummy_data)
            
            # Check feature count
            if features.shape[1] != self.n_features:
                raise ValueError(
                    f"Feature count mismatch: expected {self.n_features}, "
                    f"got {features.shape[1]}"
                )
            
            # Make prediction
            prediction = self.model.predict(features)
            
            # Check prediction format
            if not isinstance(prediction[0], (int, float, np.integer, np.floating)):
                raise ValueError(
                    f"Invalid prediction type: {type(prediction[0])}"
                )
            
            logger.info(f"✅ Model verification passed (test prediction: {prediction[0]:.2f})")
            
        except Exception as e:
            logger.error(f"Model verification failed: {e}")
            raise
    
    # ========================================================================
    # 3. PREDICTION
    # ========================================================================
    
    def predict(self, flight_data: FlightData) -> PredictionResponse:
        """
        Make prediction for a single flight.
        
        Args:
            flight_data: Validated flight input (Pydantic model)
            
        Returns:
            PredictionResponse with prediction and metadata
            
        Raises:
            RuntimeError: If models not loaded
            Exception: If prediction fails
        """
        if not ModelPredictor._models_loaded:
            raise RuntimeError(
                "Models not loaded! Call load_models() first."
            )
        
        try:
            logger.info(f"Making prediction for {flight_data.AIRLINE} "
                       f"{flight_data.ORIGIN_AIRPORT}→{flight_data.DESTINATION_AIRPORT}")
            
            # Convert Pydantic model to DataFrame
            input_df = self._prepare_input(flight_data)
            
            # Transform features using feature engineer
            features = self.feature_engineer.transform(input_df)
            
            # Verify feature count
            if features.shape[1] != self.n_features:
                raise ValueError(
                    f"Feature count mismatch: expected {self.n_features}, "
                    f"got {features.shape[1]}"
                )
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            
            # Round to 2 decimal places
            prediction = round(float(prediction), 2)
            
            logger.info(f"✅ Prediction: {prediction} {self.target_unit}")
            
            # Create response
            response = PredictionResponse(
                predicted_delay_minutes=prediction,
                model_version=self.model_version,
                timestamp=datetime.now().isoformat(),
                flight_info=create_flight_info(flight_data)
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def _prepare_input(self, flight_data: FlightData) -> pd.DataFrame:
        """
        Convert FlightData (Pydantic model) to pandas DataFrame.
        
        Args:
            flight_data: Validated flight input
            
        Returns:
            Single-row DataFrame with all required columns
        """
        # Convert to dictionary
        data_dict = {
            'MONTH': flight_data.MONTH,
            'DAY': flight_data.DAY,
            'DAY_OF_WEEK': flight_data.DAY_OF_WEEK,
            'SCHEDULED_DEPARTURE': flight_data.SCHEDULED_DEPARTURE,
            'SCHEDULED_ARRIVAL': flight_data.SCHEDULED_ARRIVAL,
            'AIRLINE': flight_data.AIRLINE,
            'ORIGIN_AIRPORT': flight_data.ORIGIN_AIRPORT,
            'DESTINATION_AIRPORT': flight_data.DESTINATION_AIRPORT
        }
        
        # Create DataFrame (single row)
        df = pd.DataFrame([data_dict])
        
        logger.debug(f"Input DataFrame shape: {df.shape}")
        logger.debug(f"Input DataFrame columns: {df.columns.tolist()}")
        
        return df
    
    # ========================================================================
    # 4. BATCH PREDICTION (Optional - for future use)
    # ========================================================================
    
    def predict_batch(self, flights: list[FlightData]) -> list[PredictionResponse]:
        """
        Make predictions for multiple flights (batch processing).
        
        More efficient than calling predict() multiple times.
        
        Args:
            flights: List of validated flight inputs
            
        Returns:
            List of prediction responses
        """
        if not ModelPredictor._models_loaded:
            raise RuntimeError("Models not loaded! Call load_models() first.")
        
        logger.info(f"Making batch prediction for {len(flights)} flights")
        
        try:
            # Convert all flights to single DataFrame
            data_dicts = [
                {
                    'MONTH': f.MONTH,
                    'DAY': f.DAY,
                    'DAY_OF_WEEK': f.DAY_OF_WEEK,
                    'SCHEDULED_DEPARTURE': f.SCHEDULED_DEPARTURE,
                    'SCHEDULED_ARRIVAL': f.SCHEDULED_ARRIVAL,
                    'AIRLINE': f.AIRLINE,
                    'ORIGIN_AIRPORT': f.ORIGIN_AIRPORT,
                    'DESTINATION_AIRPORT': f.DESTINATION_AIRPORT
                }
                for f in flights
            ]
            
            input_df = pd.DataFrame(data_dicts)
            
            # Transform features
            features = self.feature_engineer.transform(input_df)
            
            # Batch prediction
            predictions = self.model.predict(features)
            
            # Create responses
            responses = []
            for i, (flight, pred) in enumerate(zip(flights, predictions)):
                response = PredictionResponse(
                    predicted_delay_minutes=round(float(pred), 2),
                    model_version=self.model_version,
                    timestamp=datetime.now().isoformat(),
                    flight_info=create_flight_info(flight)
                )
                responses.append(response)
            
            logger.info(f"✅ Batch prediction completed: {len(responses)} results")
            
            return responses
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise
    
    # ========================================================================
    # 5. MODEL INFO
    # ========================================================================
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            "model_loaded": ModelPredictor._models_loaded,
            "model_version": self.model_version,
            "model_type": type(self.model).__name__ if self.model else None,
            "n_features": self.n_features,
            "target_name": self.target_name,
            "target_unit": self.target_unit,
            "feature_engineer_type": type(self.feature_engineer).__name__ if self.feature_engineer else None,
            "data_validator_loaded": self.data_validator is not None
        }
    
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return ModelPredictor._models_loaded


# ============================================================================
# 6. CONVENIENCE FUNCTIONS - Easy API for main.py
# ============================================================================

# Global predictor instance (singleton)
_predictor = None


def get_predictor() -> ModelPredictor:
    """
    Get or create global predictor instance.
    
    This is the recommended way to access the predictor in FastAPI.
    
    Returns:
        ModelPredictor singleton instance
        
    Example:
        predictor = get_predictor()
        if not predictor.is_loaded():
            predictor.load_models()
    """
    global _predictor
    if _predictor is None:
        _predictor = ModelPredictor()
    return _predictor


def predict_flight(flight_data: FlightData) -> PredictionResponse:
    """
    Convenience function for single prediction.
    
    Args:
        flight_data: Validated flight input
        
    Returns:
        Prediction response
        
    Example:
        from app.models import FlightData
        from app.predictor import predict_flight
        
        flight = FlightData(MONTH=6, DAY=15, ...)
        result = predict_flight(flight)
        print(result.predicted_delay_minutes)
    """
    predictor = get_predictor()
    return predictor.predict(flight_data)


# ============================================================================
# 7. TEST CODE
# ============================================================================

if __name__ == "__main__":
    # Fix imports for standalone testing
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*70)
    print("TESTING MODEL PREDICTOR")
    print("="*70)
    
    # ========================================================================
    # Test 1: Model Loading
    # ========================================================================
    print("\n1.  Testing model loading...")
    
    try:
        predictor = ModelPredictor()
        predictor.load_models()
        print("   ✅ Models loaded successfully!")
        
        # Print model info
        info = predictor.get_model_info()
        print("\n   Model Info:")
        for key, value in info.items():
            print(f"      {key}: {value}")
        
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        sys.exit(1)
    
    # ========================================================================
    # Test 2: Single Prediction
    # ========================================================================
    print("\n2.  Testing single prediction...")
    
    try:
        from app.models import FlightData
        
        # Create test flight
        test_flight = FlightData(
            MONTH=6,
            DAY=15,
            DAY_OF_WEEK=2,
            SCHEDULED_DEPARTURE=1420,
            SCHEDULED_ARRIVAL=1650,
            AIRLINE="AA",
            ORIGIN_AIRPORT="ORD",
            DESTINATION_AIRPORT="LAX"
        )
        
        # Make prediction
        result = predictor.predict(test_flight)
        
        print("   ✅ Prediction successful!")
        print(f"\n   Result:")
        print(f"      Predicted delay: {result.predicted_delay_minutes} minutes")
        print(f"      Model version: {result.model_version}")
        print(f"      Flight: {result.flight_info['route']}")
        
    except Exception as e:
        print(f"   ❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # Test 3: Multiple Predictions (same predictor instance)
    # ========================================================================
    print("\n3.  Testing multiple predictions...")
    
    test_flights = [
        ("DL", "ATL", "LAX"),
        ("UA", "ORD", "SFO"),
        ("WN", "LAS", "PHX")
    ]
    
    for airline, origin, dest in test_flights:
        try:
            flight = FlightData(
                MONTH=7,
                DAY=20,
                DAY_OF_WEEK=3,
                SCHEDULED_DEPARTURE=1000,
                SCHEDULED_ARRIVAL=1300,
                AIRLINE=airline,
                ORIGIN_AIRPORT=origin,
                DESTINATION_AIRPORT=dest
            )
            
            result = predictor.predict(flight)
            print(f"   ✅ {airline} {origin}→{dest}: {result.predicted_delay_minutes:.1f} min")
            
        except Exception as e:
            print(f"   ❌ {airline} {origin}→{dest}: {e}")
    
    # ========================================================================
    # Test 4: Batch Prediction
    # ========================================================================
    print("\n4.  Testing batch prediction...")
    
    try:
        batch_flights = [
            FlightData(
                MONTH=i % 12 + 1,
                DAY=15,
                DAY_OF_WEEK=2,
                SCHEDULED_DEPARTURE=1000 + i*100,
                SCHEDULED_ARRIVAL=1300 + i*100,
                AIRLINE=["AA", "DL", "UA"][i % 3],
                ORIGIN_AIRPORT="ORD",
                DESTINATION_AIRPORT="LAX"
            )
            for i in range(5)
        ]
        
        results = predictor.predict_batch(batch_flights)
        
        print(f"   ✅ Batch prediction successful! ({len(results)} flights)")
        for i, result in enumerate(results):
            print(f"      Flight {i+1}: {result.predicted_delay_minutes:.1f} min")
        
    except Exception as e:
        print(f"   ❌ Batch prediction failed: {e}")
    
    # ========================================================================
    # Test 5: Singleton Pattern Verification
    # ========================================================================
    print("\n5.  Testing singleton pattern...")
    
    predictor2 = ModelPredictor()
    predictor3 = get_predictor()
    
    if predictor is predictor2 is predictor3:
        print("   ✅ Singleton pattern verified!")
        print(f"      All instances are the same object (id: {id(predictor)})")
    else:
        print("   ❌ Singleton pattern broken!")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETED!")
    print("="*70 + "\n")