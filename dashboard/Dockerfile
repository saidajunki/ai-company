FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy source and static files
COPY src/ src/
COPY static/ static/

# Set PYTHONPATH so bare imports work
ENV PYTHONPATH=/app/src

# Default environment variables
ENV DATA_DIR=/app/data
ENV COMPANY_ID=alpha
ENV POLL_INTERVAL=10

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Expose port
EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app/src"]
