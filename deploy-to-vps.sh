#!/bin/bash

# AI Company VPS Deployment Script
# Deploys the ai-company service directly on VPS (systemd)

set -e

VPS_HOST="BableApp"
APP_PATH="/opt/apps/ai-company"
SERVICE_NAME="ai-company"

echo "🚀 Deploying AI Company to VPS..."
echo "   Host: $VPS_HOST"
echo "   App Path: $APP_PATH"
echo "   Service: $SERVICE_NAME"
echo ""

# Step 1: Pull latest code
echo "📦 Step 1: Syncing repository on VPS..."
ssh "$VPS_HOST" bash << 'DEPLOY_SCRIPT'
  APP_PATH="/opt/apps/ai-company"
  if [ -d "$APP_PATH" ]; then
    echo "   Repository exists, pulling latest changes..."
    cd "$APP_PATH"
    git pull origin main
  else
    echo "   Cloning repository..."
    mkdir -p /opt/apps
    cd /opt/apps
    git clone git@github-ai-company:saidajunki/ai-company.git ai-company
  fi
DEPLOY_SCRIPT

# Step 2: Set up environment variables
echo "🔑 Step 2: Checking environment variables..."
ssh "$VPS_HOST" bash << 'DEPLOY_SCRIPT'
  APP_PATH="/opt/apps/ai-company"
  cd "$APP_PATH"
  if [ ! -f .env ]; then
    if [ -f .env.template ]; then
      echo "   ⚠ .env not found — creating from template"
      cp .env.template .env
      chmod 600 .env
      echo "   Please set secrets in $APP_PATH/.env"
    else
      echo "   ⚠ .env and .env.template not found"
    fi
  else
    echo "   ✓ .env file exists"
  fi
DEPLOY_SCRIPT

# Step 3: Update dependencies and restart service
echo "🔄 Step 3: Updating dependencies and restarting service..."
ssh "$VPS_HOST" bash << 'DEPLOY_SCRIPT'
  APP_PATH="/opt/apps/ai-company"
  cd "$APP_PATH"
  .venv/bin/pip install -q "."
  systemctl restart ai-company
  echo "   ✓ Service restarted"
DEPLOY_SCRIPT

# Step 4: Verify
echo "⏳ Step 4: Verifying..."
sleep 3

ssh "$VPS_HOST" bash << 'DEPLOY_SCRIPT'
  if systemctl is-active --quiet ai-company; then
    echo "   ✓ ai-company service is running"
  else
    echo "   ⚠ ai-company service is NOT running"
    journalctl -u ai-company --no-pager -n 10
  fi
DEPLOY_SCRIPT

echo ""
echo "✨ Deployment complete!"
echo ""
echo "To check status:"
echo "   ssh $VPS_HOST 'systemctl status ai-company'"
echo "   ssh $VPS_HOST 'journalctl -u ai-company -f'"
