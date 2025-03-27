# Build and test stage
FROM python:3.12-slim as builder

WORKDIR /app

# Copy requirements files
COPY requirements.txt requirements-dev.txt ./

# Install all dependencies (including dev dependencies)
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy source code and tests
COPY src/ ./src/
COPY tests/ ./tests/

# Run tests
RUN pytest

# Production stage
FROM python:3.12-slim

WORKDIR /app

# Copy only production requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the source code (no tests)
COPY src/ ./src/

# Run the application
CMD uvicorn src.howdoyoufindme.main:app --host 0.0.0.0 --port=${PORT:-8080}
