# Multi-stage Dockerfile for Aerodrome AI Agent
# Optimized for production deployment on Google Cloud Run

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILDPLATFORM
ARG TARGETPLATFORM

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment and install Python dependencies
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Set build info
ARG BUILD_DATE
ARG GIT_COMMIT
ARG VERSION
LABEL build-date=$BUILD_DATE
LABEL git-commit=$GIT_COMMIT
LABEL version=$VERSION

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY src/ ./src/
COPY *.py ./

# Create necessary directories
RUN mkdir -p /app/logs /app/static /app/temp && \
    chown -R appuser:appuser /app

# Copy static files for error pages
COPY static/ ./static/

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Default environment variables
ENV PORT=8080 \
    HOST=0.0.0.0 \
    ENV=production \
    LOG_LEVEL=INFO \
    WORKERS=1 \
    MAX_REQUESTS=1000 \
    MAX_REQUESTS_JITTER=50 \
    TIMEOUT=120 \
    GRACEFUL_TIMEOUT=30

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health/live || exit 1

# Switch to non-root user
USER appuser

# Expose port
EXPOSE $PORT

# Production startup command
CMD exec uvicorn src.api.main:app \
    --host $HOST \
    --port $PORT \
    --workers $WORKERS \
    --loop uvloop \
    --http httptools \
    --log-level $LOG_LEVEL \
    --access-log \
    --no-server-header \
    --no-date-header \
    --timeout-keep-alive 5 \
    --timeout-graceful-shutdown $GRACEFUL_TIMEOUT