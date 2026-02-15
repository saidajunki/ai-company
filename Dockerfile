FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY tests/ tests/

# Install Python dependencies
RUN pip install --no-cache-dir ".[dev]"

# Create non-root user
RUN useradd -m -u 1000 company
USER company

# Default: run tests to verify installation
CMD ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]
