"""
Pydantic models for request/response validation.

These models define the API contract - what data we accept and return.
FastAPI uses these models to:
- Automatically validate incoming requests
- Generate JSON schema for API documentation
- Serialize responses to JSON
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, Dict, Any
from datetime import datetime

# Global Pydantic config - disable protected namespace warnings
PYDANTIC_CONFIG = ConfigDict(protected_namespaces=())

# ============================================================================
# 1. INPUT MODEL - What API receives from user
# ============================================================================

class FlightData(BaseModel):
    """
    Flight data input for prediction.
    
    This is the RAW data that user sends - human-readable format.
    The AirportFeatureEngineer will transform it into model features.
    
    Example request:
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
    """
    
    # Temporal features
    MONTH: int = Field(
        ...,  # ... means REQUIRED field
        ge=1,  # ge = greater or equal
        le=12,  # le = less or equal
        description="Month of flight (1-12)",
        examples=[6]
    )
    
    DAY: int = Field(
        ...,
        ge=1,
        le=31,
        description="Day of month (1-31)",
        examples=[15]
    )
    
    DAY_OF_WEEK: int = Field(
        ...,
        ge=1,
        le=7,
        description="Day of week (1=Monday, 7=Sunday)",
        examples=[2]
    )
    
    # Time features (HHMM format: 1420 = 2:20 PM)
    SCHEDULED_DEPARTURE: int = Field(
        ...,
        ge=0,
        le=2359,
        description="Scheduled departure time in HHMM format (e.g., 1420 = 2:20 PM)",
        examples=[1420]
    )
    
    SCHEDULED_ARRIVAL: int = Field(
        ...,
        ge=0,
        le=2359,
        description="Scheduled arrival time in HHMM format",
        examples=[1650]
    )
    
    # Airline (2-letter code)
    AIRLINE: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Airline code (e.g., 'AA', 'DL', 'UA')",
        examples=["AA"]
    )
    
    # Airports (3-letter codes)
    ORIGIN_AIRPORT: str = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Origin airport code (e.g., 'ORD', 'ATL')",
        examples=["ORD"]
    )
    
    DESTINATION_AIRPORT: str = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Destination airport code",
        examples=["LAX"]
    )
    
    # ========================================================================
    # CUSTOM VALIDATORS - Advanced validation logic
    # ========================================================================
    
    @field_validator('AIRLINE')
    @classmethod
    def validate_airline_code(cls, v: str) -> str:
        """
        Validate and normalize airline code.
        
        - Converts to uppercase
        - Checks against known airlines (optional)
        
        Args:
            v: Airline code from user input
            
        Returns:
            Uppercase airline code
            
        Raises:
            ValueError: If airline code is unknown
        """
        # Always convert to uppercase
        v = v.upper()
        
        # Optional: Check against known airlines
        from app.config import VALID_AIRLINES
        
        if VALID_AIRLINES is not None:
            if v not in VALID_AIRLINES:
                raise ValueError(
                    f"Unknown airline code: '{v}'. "
                    f"Valid codes: {', '.join(VALID_AIRLINES)}"
                )
        
        return v
    
    @field_validator('ORIGIN_AIRPORT', 'DESTINATION_AIRPORT')
    @classmethod
    def validate_airport_code(cls, v: str) -> str:
        """
        Validate and normalize airport code.
        
        - Converts to uppercase
        - Ensures 3-letter format
        
        Args:
            v: Airport code from user input
            
        Returns:
            Uppercase airport code
        """
        return v.upper()
    
    @field_validator('SCHEDULED_DEPARTURE', 'SCHEDULED_ARRIVAL')
    @classmethod
    def validate_time_format(cls, v: int, info) -> int:
        """
        Validate HHMM time format.
        
        Checks:
        - Hours: 0-23
        - Minutes: 0-59
        
        Examples:
            1420 → Valid (14:20)
            830  → Valid (08:30)
            2570 → Invalid (25:70 doesn't exist!)
            
        Args:
            v: Time in HHMM format
            info: Field info (used for error messages)
            
        Returns:
            Validated time value
            
        Raises:
            ValueError: If time format is invalid
        """
        hours = v // 100
        minutes = v % 100
        
        # Validate hours (0-23)
        if hours < 0 or hours > 23:
            raise ValueError(
                f"{info.field_name} has invalid hours: {hours} "
                f"(must be 0-23). Got: {v}"
            )
        
        # Validate minutes (0-59)
        if minutes < 0 or minutes > 59:
            raise ValueError(
                f"{info.field_name} has invalid minutes: {minutes} "
                f"(must be 0-59). Got: {v}"
            )
        
        return v
    
    # Model configuration
    model_config = {
        **PYDANTIC_CONFIG,
        "json_schema_extra": {
            "examples": [
                {
                    "MONTH": 6,
                    "DAY": 15,
                    "DAY_OF_WEEK": 2,
                    "SCHEDULED_DEPARTURE": 1420,
                    "SCHEDULED_ARRIVAL": 1650,
                    "DISTANCE": 1745,
                    "AIRLINE": "AA",
                    "ORIGIN_AIRPORT": "ORD",
                    "DESTINATION_AIRPORT": "LAX"
                }
            ]
        }
    }


# ============================================================================
# 2. OUTPUT MODEL - Success response
# ============================================================================

class PredictionResponse(BaseModel):
    """
    Successful prediction response.
    
    Returned when prediction succeeds.
    
    Example:
        {
            "predicted_delay_minutes": 12.5,
            "model_version": "random_forest",
            "timestamp": "2025-02-09T14:23:45.123456",
            "flight_info": {
                "airline": "AA",
                "route": "ORD → LAX",
                "scheduled_departure": "14:20",
                "distance_miles": 1745
            }
        }
    """
    
    predicted_delay_minutes: float = Field(
        ...,
        description="Predicted arrival delay in minutes (negative = early)",
        examples=[12.5]
    )
    
    model_version: str = Field(
        ...,
        description="Model version used for prediction",
        examples=["random_forest"]
    )
    
    timestamp: str = Field(
        ...,
        description="Prediction timestamp (ISO format)",
        examples=["2025-02-09T14:23:45.123456"]
    )
    
    flight_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional flight information for context",
        examples=[{
            "airline": "AA",
            "route": "ORD → LAX",
            "scheduled_departure": "14:20",
            "scheduled_arrival": "16:50",
            "distance_miles": 1745
        }]
    )
    
    model_config = {
        **PYDANTIC_CONFIG,
        "json_schema_extra": {
            "examples": [
                {
                    "predicted_delay_minutes": 12.5,
                    "model_version": "random_forest",
                    "timestamp": "2025-02-09T14:23:45.123456",
                    "flight_info": {
                        "airline": "AA",
                        "route": "ORD → LAX"
                    }
                }
            ]
        }
    }


# ============================================================================
# 3. ERROR MODEL - Error response
# ============================================================================

class ErrorResponse(BaseModel):
    """
    Error response for failed requests.
    
    Returned when validation fails or prediction errors occur.
    
    Example:
        {
            "error": "ValidationError",
            "message": "Invalid input data",
            "details": {
                "field": "MONTH",
                "value": 13,
                "constraint": "Must be between 1 and 12"
            },
            "timestamp": "2025-02-09T14:23:45.123456"
        }
    """
    
    error: str = Field(
        ...,
        description="Error type",
        examples=["ValidationError"]
    )
    
    message: str = Field(
        ...,
        description="Human-readable error message",
        examples=["Invalid input data"]
    )
    
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details",
        examples=[{
            "field": "MONTH",
            "value": 13,
            "constraint": "Must be between 1 and 12"
        }]
    )
    
    timestamp: str = Field(
        ...,
        description="Error timestamp (ISO format)",
        examples=["2025-02-09T14:23:45.123456"]
    )

    model_config = PYDANTIC_CONFIG

# ============================================================================
# 4. HEALTH CHECK MODEL
# ============================================================================

class HealthResponse(BaseModel):
    """
    Health check response.
    
    Returned by /health endpoint to check API status.
    
    Example:
        {
            "status": "healthy",
            "model_loaded": true,
            "version": "v1",
            "model_version": "random_forest"
        }
    """
    
    status: str = Field(
        ...,
        description="API health status ('healthy' or 'unhealthy')",
        examples=["healthy"]
    )
    
    model_loaded: bool = Field(
        ...,
        description="Whether ML model is loaded in memory",
        examples=[True]
    )
    
    version: str = Field(
        ...,
        description="API version",
        examples=["v1"]
    )
    
    model_version: str = Field(
        ...,
        description="Loaded model version",
        examples=["random_forest"]
    )

    model_config = PYDANTIC_CONFIG

# ============================================================================
# 5. HELPER FUNCTIONS
# ============================================================================

def format_time_hhmm(time_int: int) -> str:
    """
    Convert HHMM integer to readable time string.
    
    Args:
        time_int: Time in HHMM format (e.g., 1420)
        
    Returns:
        Formatted time string (e.g., "14:20")
        
    Examples:
        >>> format_time_hhmm(1420)
        '14:20'
        >>> format_time_hhmm(830)
        '08:30'
        >>> format_time_hhmm(45)
        '00:45'
    """
    hours = time_int // 100
    minutes = time_int % 100
    return f"{hours:02d}:{minutes:02d}"


def create_flight_info(flight_data: FlightData) -> Dict[str, Any]:
    """
    Create user-friendly flight info dict from FlightData.
    
    Useful for adding context to prediction response.
    
    Args:
        flight_data: Validated flight input
        
    Returns:
        Flight info dictionary with formatted fields
        
    Example:
        >>> flight = FlightData(MONTH=6, DAY=15, ...)
        >>> create_flight_info(flight)
        {
            "airline": "AA",
            "route": "ORD → LAX",
            "scheduled_departure": "14:20",
            "scheduled_arrival": "16:50",
        }
    """
    return {
        "airline": flight_data.AIRLINE,
        "route": f"{flight_data.ORIGIN_AIRPORT} → {flight_data.DESTINATION_AIRPORT}",
        "scheduled_departure": format_time_hhmm(flight_data.SCHEDULED_DEPARTURE),
        "scheduled_arrival": format_time_hhmm(flight_data.SCHEDULED_ARRIVAL),
    }


# ============================================================================
# 6. TEST CODE - Run this file directly to test models
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING PYDANTIC MODELS")
    print("="*70)
    
    # ========================================================================
    # Test 1: Valid FlightData
    # ========================================================================
    print("\n1. Testing VALID FlightData...")
    
    valid_data = {
        "MONTH": 6,
        "DAY": 15,
        "DAY_OF_WEEK": 2,
        "SCHEDULED_DEPARTURE": 1420,
        "SCHEDULED_ARRIVAL": 1650,
        "DISTANCE": 1745,
        "AIRLINE": "aa",  # lowercase - should convert to "AA"
        "ORIGIN_AIRPORT": "ord",  # lowercase - should convert to "ORD"
        "DESTINATION_AIRPORT": "LAX"
    }
    
    try:
        flight = FlightData(**valid_data)
        print(f"   ✅ Valid data accepted!")
        print(f"   Airline (normalized): {flight.AIRLINE}")
        print(f"   Origin (normalized): {flight.ORIGIN_AIRPORT}")
        print(f"\n   JSON representation:")
        print(f"   {flight.model_dump_json(indent=2)}")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
    
    # ========================================================================
    # Test 2: Invalid FlightData - Multiple errors
    # ========================================================================
    print("\n2. Testing INVALID FlightData (should fail)...")
    
    invalid_data = {
        "MONTH": 13,  # Out of range (1-12)
        "DAY": 15,
        "DAY_OF_WEEK": 2,
        "SCHEDULED_DEPARTURE": 2570,  # Invalid time (25:70)
        "SCHEDULED_ARRIVAL": 1650,
        "DISTANCE": -100,  # Negative distance
        "AIRLINE": "ZZZ",  # 3 letters (invalid)
        "ORIGIN_AIRPORT": "ORD",
        "DESTINATION_AIRPORT": "LAX"
    }
    
    try:
        flight = FlightData(**invalid_data)
        print(f"   ❌ ERROR: Invalid data was accepted (shouldn't happen!)")
    except Exception as e:
        print(f"   ✅ Validation correctly rejected invalid data!")
        print(f"\n   Errors caught:")
        # Pydantic wraps errors - print them nicely
        error_lines = str(e).split('\n')
        for line in error_lines[:5]:  # First 5 lines
            print(f"      {line}")
    
    # ========================================================================
    # Test 3: Time format validation
    # ========================================================================
    print("\n3. Testing time format validation...")
    
    test_times = [
        (1420, True, "14:20 - valid"),
        (830, True, "08:30 - valid"),
        (2570, False, "25:70 - invalid hours and minutes"),
        (1399, False, "13:99 - invalid minutes"),
    ]
    
    for time_val, should_pass, description in test_times:
        try:
            test_data = valid_data.copy()
            test_data["SCHEDULED_DEPARTURE"] = time_val
            flight = FlightData(**test_data)
            if should_pass:
                print(f"   ✅ {description} - passed")
            else:
                print(f"   ❌ {description} - should have failed!")
        except ValueError as e:
            if not should_pass:
                print(f"   ✅ {description} - correctly rejected")
            else:
                print(f"   ❌ {description} - incorrectly rejected: {e}")
    
    # ========================================================================
    # Test 4: PredictionResponse
    # ========================================================================
    print("\n4. Testing PredictionResponse...")
    
    # Create response using helper function
    flight = FlightData(**valid_data)
    flight_info = create_flight_info(flight)
    
    response = PredictionResponse(
        predicted_delay_minutes=12.5,
        model_version="random_forest",
        timestamp=datetime.now().isoformat(),
        flight_info=flight_info
    )
    
    print(f"   ✅ Response created successfully!")
    print(f"\n   JSON representation:")
    print(f"   {response.model_dump_json(indent=2)}")
    
    # ========================================================================
    # Test 5: Helper functions
    # ========================================================================
    print("\n5. Testing helper functions...")
    
    test_times_format = [1420, 830, 45, 2359]
    for t in test_times_format:
        formatted = format_time_hhmm(t)
        print(f"   {t:4d} → {formatted}")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETED!")
    print("="*70 + "\n")