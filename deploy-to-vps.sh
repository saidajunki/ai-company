#!/bin/bash

# AI Company VPS Deployment Script
# Deploys the ai-company container to VPS

set -e

VPS_HOST="BableApp"
APP_PATH="/opt/apps/ai-company"
SERVICE_NAME="ai-company"

echo "🚀 Deploying AI Company to VPS..."
echo "   Host: $VPS_HOST"
echo "   App Path: $APP_PATH"
echo "   Service: $SERVICE_NAME"
echo ""

# Step 1: Clone or pull repository
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
echo "🔑 Step 2: Setting up environment variables..."
ssh "$VPS_HOST" bash << 'DEPLOY_SCRIPT'
  APP_PATH="/opt/apps/ai-company"
  cd "$APP_PATH"
  # Create .env if it doesn't exist (tokens must be set manually or via this script)
  if [ ! -f .env ]; then
    echo "   ⚠ .env file not found – creating placeholder"
    echo "SLACK_APP_TOKEN=" > .env
    echo "SLACK_BOT_TOKEN=" >> .env
    echo "   Please set SLACK_APP_TOKEN and SLACK_BOT_TOKEN in $APP_PATH/.env"
  else
    echo "   ✓ .env file exists"
  fi
DEPLOY_SCRIPT

# Step 3: Build and run container
echo "🐳 Step 3: Building and starting Docker container..."
ssh "$VPS_HOST" bash << 'DEPLOY_SCRIPT'
  APP_PATH="/opt/apps/ai-company"
  cd "$APP_PATH"
  docker compose build
  docker compose up -d
  echo "   ✓ Container started"
DEPLOY_SCRIPT

# Step 4: Wait and verify
echo "⏳ Step 4: Verifying..."
sleep 3

ssh "$VPS_HOST" bash << 'DEPLOY_SCRIPT'
  docker ps -a | grep ai-company || echo "   ⚠ Container not found in docker ps"
  echo ""
  echo "   Checking logs..."
  docker logs ai-company --tail 20 2>&1 || true
DEPLOY_SCRIPT

echo ""
echo "✨ Deployment complete!"
echo ""
echo "To check status:"
echo "   ssh $VPS_HOST 'docker logs ai-company'"
