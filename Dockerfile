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

# Create non-root user and data directory with correct ownership
RUN useradd -m -u 1000 company \
    && mkdir -p /app/data \
    && chown -R company:company /app/data
USER company

# Health check via heartbeat file existence
HEALTHCHECK --interval=60s --timeout=5s --retries=3 \
    CMD test -f /app/data/companies/${COMPANY_ID:-alpha}/state/heartbeat.json || exit 1

# Run the manager as a long-running process
CMD ["python", "-m", "main"]
