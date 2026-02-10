# ============================================================================
# Flight Delay Prediction API - Dockerfile
# ============================================================================
# 
# This Dockerfile creates a containerized FastAPI application for flight 
# delay predictions.
#
# Build:
#   docker build -t flight-delay-api .
#
# Run:
#   docker run -p 8000:8000 flight-delay-api
#
# ============================================================================

# ----------------------------------------------------------------------------
# STAGE 1: Base Image
# ----------------------------------------------------------------------------
# Use official Python 3.12 slim image (Debian-based, smaller than full)
FROM python:3.12-slim

# Set metadata labels
LABEL maintainer="your-email@example.com"
LABEL description="Flight Delay Prediction API"
LABEL version="1.0"

# ----------------------------------------------------------------------------
# STAGE 2: System Setup
# ----------------------------------------------------------------------------

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN groupadd -r apiuser && \
    useradd -r -g apiuser apiuser && \
    mkdir -p /app/logs && \
    chown -R apiuser:apiuser /app

# ----------------------------------------------------------------------------
# STAGE 3: Install Dependencies
# ----------------------------------------------------------------------------

# Copy requirements file first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------------------------------------------------------
# STAGE 4: Copy Application Code
# ----------------------------------------------------------------------------

# Copy application code
COPY app/ ./app/
COPY src/ ./src/

# Copy model files (these are large - should be last layer)
COPY models/ ./models/

# ----------------------------------------------------------------------------
# STAGE 5: Configuration
# ----------------------------------------------------------------------------

# Switch to non-root user
USER apiuser

# Expose port (documentation only, doesn't actually publish)
EXPOSE 8000

# Health check (Docker will ping this endpoint)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# ----------------------------------------------------------------------------
# STAGE 6: Startup Command
# ----------------------------------------------------------------------------

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]