"""
FastAPI Application - Flight Delay Prediction API.

This is the main entry point for the API. It:
- Initializes FastAPI app
- Loads ML models on startup
- Defines API endpoints (/health, /predict)
- Handles errors gracefully
- Provides auto-generated API documentation (Swagger)

Run:
    uvicorn app.main:app --reload
    
Visit:
    http://localhost:8000/docs (Swagger UI)
    http://localhost:8000/redoc (ReDoc)
"""

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from datetime import datetime

from app.config import settings, MODEL_VERSION, print_config
from app.models import (
    FlightData, 
    PredictionResponse, 
    ErrorResponse,
    HealthResponse
)
from app.predictor import get_predictor


# ============================================================================
# 1. LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console
        logging.FileHandler(settings.LOG_FILE)  # File
    ]
)

logger = logging.getLogger(__name__)


# ============================================================================
# 2. LIFESPAN EVENTS - Startup & Shutdown
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    
    Startup:
        - Load ML models into memory
        - Print configuration
        
    Shutdown:
        - Cleanup (if needed)
    """
    # STARTUP
    logger.info("="*70)
    logger.info("STARTING FLIGHT DELAY PREDICTION API")
    logger.info("="*70)
    
    # Print configuration
    if settings.DEBUG:
        print_config()
    
    # Load models
    if settings.LOAD_MODEL_ON_STARTUP:
        logger.info("Loading ML models on startup...")
        try:
            predictor = get_predictor()
            predictor.load_models()
            logger.info("✅ Models loaded successfully!")
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            logger.error("   API will start but predictions will fail!")
    else:
        logger.info("⚠️  Model loading skipped (lazy loading enabled)")
    
    logger.info("="*70)
    logger.info("API READY!")
    logger.info(f"Docs: http://{settings.API_HOST}:{settings.API_PORT}/docs")
    logger.info("="*70)
    
    yield  # App runs here
    
    # SHUTDOWN
    logger.info("Shutting down API...")


# ============================================================================
# 3. FASTAPI APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="Flight Delay Prediction API",
    description=(
        "Predict flight arrival delays using machine learning. "
        "Trained on 4.5M US domestic flights (2015)."
    ),
    version=settings.API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# ============================================================================
# 4. MIDDLEWARE - CORS (Optional)
# ============================================================================

if settings.ALLOW_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info(f"✅ CORS enabled for origins: {settings.CORS_ORIGINS}")


# ============================================================================
# 5. EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with custom error response."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.__class__.__name__,
            message=exc.detail,
            timestamp=datetime.now().isoformat()
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    
    # In production, don't expose internal errors
    if settings.DEBUG:
        error_detail = str(exc)
    else:
        error_detail = "Internal server error"
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message=error_detail,
            timestamp=datetime.now().isoformat()
        ).model_dump()
    )


# ============================================================================
# 6. API ENDPOINTS
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - API information.
    
    Returns basic API info and links to documentation.
    """
    return {
        "name": "Flight Delay Prediction API",
        "version": settings.API_VERSION,
        "model_version": MODEL_VERSION,
        "status": "operational",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "predict": "/predict"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with API status and model loading state
        
    Example:
        GET /health
        
        Response:
        {
            "status": "healthy",
            "model_loaded": true,
            "version": "v1",
            "model_version": "random_forest"
        }
    """
    predictor = get_predictor()
    
    return HealthResponse(
        status="healthy" if predictor.is_loaded() else "degraded",
        model_loaded=predictor.is_loaded(),
        version=settings.API_VERSION,
        model_version=MODEL_VERSION
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"]
)
async def predict_delay(flight_data: FlightData):
    """
    Predict flight arrival delay.
    
    Makes a prediction for a single flight based on input features.
    
    Args:
        flight_data: Flight information (validated by Pydantic)
        
    Returns:
        PredictionResponse with predicted delay and metadata
        
    Raises:
        HTTPException 503: If model not loaded
        HTTPException 422: If input validation fails (automatic)
        HTTPException 500: If prediction fails
        
    Example:
        POST /predict
        
        Request:
        {
            "MONTH": 6,
            "DAY": 15,
            "DAY_OF_WEEK": 2,
            "SCHEDULED_DEPARTURE": 1420,
            "SCHEDULED_ARRIVAL": 1650,
            "AIRLINE": "AA",
            "ORIGIN_AIRPORT": "ORD",
            "DESTINATION_AIRPORT": "LAX"
        }
        
        Response:
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
    """
    # Get predictor instance
    predictor = get_predictor()
    
    # Check if model is loaded
    if not predictor.is_loaded():
        logger.error("Prediction request received but model not loaded!")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later."
        )
    
    # Log request
    logger.info(
        f"Prediction request: {flight_data.AIRLINE} "
        f"{flight_data.ORIGIN_AIRPORT}→{flight_data.DESTINATION_AIRPORT}"
    )
    
    try:
        # Make prediction
        result = predictor.predict(flight_data)
        
        # Log result
        logger.info(
            f"Prediction: {result.predicted_delay_minutes:.2f} minutes "
            f"for {flight_data.AIRLINE} "
            f"{flight_data.ORIGIN_AIRPORT}→{flight_data.DESTINATION_AIRPORT}"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


# ============================================================================
# 7. OPTIONAL - Batch Prediction Endpoint
# ============================================================================

# Uncomment this if you want batch prediction support

# from typing import List

# @app.post(
#     "/predict/batch",
#     response_model=List[PredictionResponse],
#     tags=["Prediction"]
# )
# async def predict_batch(flights: List[FlightData]):
#     """
#     Predict delays for multiple flights (batch processing).
    
#     More efficient than multiple /predict calls.
    
#     Args:
#         flights: List of flight data (max 100)
        
#     Returns:
#         List of prediction responses
#     """
#     # Limit batch size
#     if len(flights) > 100:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Batch size too large (max 100 flights)"
#         )
    
#     predictor = get_predictor()
    
#     if not predictor.is_loaded():
#         raise HTTPException(
#             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
#             detail="Model not loaded"
#         )
    
#     try:
#         results = predictor.predict_batch(flights)
#         logger.info(f"Batch prediction: {len(results)} flights processed")
#         return results
#     except Exception as e:
#         logger.error(f"Batch prediction failed: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Batch prediction failed: {str(e)}"
#         )


# ============================================================================
# 8. DEV/DEBUG ENDPOINTS (Only in DEBUG mode)
# ============================================================================

if settings.DEBUG:
    @app.get("/debug/config", tags=["Debug"])
    async def debug_config():
        """Show current configuration (DEBUG mode only)."""
        return {
            "settings": {
                "API_HOST": settings.API_HOST,
                "API_PORT": settings.API_PORT,
                "API_VERSION": settings.API_VERSION,
                "DEBUG": settings.DEBUG,
                "LOG_LEVEL": settings.LOG_LEVEL,
            },
            "model": {
                "version": MODEL_VERSION,
                "loaded": get_predictor().is_loaded()
            }
        }
    
    @app.get("/debug/model-info", tags=["Debug"])
    async def debug_model_info():
        """Get detailed model information (DEBUG mode only)."""
        predictor = get_predictor()
        if not predictor.is_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        return predictor.get_model_info()


# ============================================================================
# 9. MAIN - For direct execution (development)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting development server...")
    logger.info(f"Running on http://{settings.API_HOST}:{settings.API_PORT}")
    
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,  # Auto-reload on code changes
        log_level=settings.LOG_LEVEL.lower()
    )