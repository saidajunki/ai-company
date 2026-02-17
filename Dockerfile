FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install Python 3.12 + common system tools + Docker CLI
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    curl \
    wget \
    git \
    sudo \
    jq \
    dnsutils \
    net-tools \
    ca-certificates \
    gnupg \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && chmod a+r /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y docker-ce-cli docker-compose-plugin \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY tests/ tests/

# Make python3 available as python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Install Python dependencies
RUN pip install --no-cache-dir --break-system-packages ".[dev]"

# Create non-root user and data directory with correct ownership
# Ubuntu base image has a default 'ubuntu' user at UID 1000, remove it first
RUN userdel -r ubuntu 2>/dev/null || true \
    && groupadd -g 987 docker \
    && useradd -m -u 1000 -G docker company \
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
