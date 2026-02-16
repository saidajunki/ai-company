FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY tests/ tests/

# Install Python dependencies
RUN pip install --no-cache-dir ".[dev]"

# Create non-root user and data directory with correct ownership
RUN useradd -m -u 1000 company \
    && mkdir -p /etc/sudoers.d \
    && echo "company ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/company \
    && chmod 0440 /etc/sudoers.d/company \
    && mkdir -p /app/data \
    && chown -R company:company /app/data
USER company

# Health check via heartbeat freshness
HEALTHCHECK --interval=60s --timeout=5s --retries=3 \
    CMD python -c "import json,os,sys;from datetime import datetime,timezone,timedelta;cid=os.environ.get('COMPANY_ID','alpha');p=f'/app/data/companies/{cid}/state/heartbeat.json';m=int(os.environ.get('HEALTHCHECK_STALE_MINUTES','20'));d=json.load(open(p,'r',encoding='utf-8'));ts=str(d.get('updated_at',''));ts=ts[:-1]+'+00:00' if ts.endswith('Z') else ts;u=datetime.fromisoformat(ts);u=u.replace(tzinfo=timezone.utc) if u.tzinfo is None else u;sys.exit(0 if datetime.now(timezone.utc)-u<=timedelta(minutes=m) else 1)"

# Run the manager as a long-running process
CMD ["python", "-m", "supervisor"]
