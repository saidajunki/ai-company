#!/bin/bash
# install-watchdog.sh — Watchdog の cron ジョブをインストールする
#
# 使い方:
#   1. 環境変数を設定する（下記参照）
#   2. chmod +x install-watchdog.sh
#   3. ./install-watchdog.sh
#
# ===== 必須環境変数 =====
#
# HEARTBEAT_PATH : heartbeat.json のフルパス
#   例: /opt/apps/ai-company/companies/default/state/heartbeat.json
#
# SLACK_WEBHOOK_URL : Slack Incoming Webhook の URL
#   例: https://hooks.slack.com/services/YOUR/WEBHOOK/URL
#
# ===== オプション環境変数 =====
#
# CONTAINER_NAME : Docker コンテナ名（デフォルト: ai-company）
#   例: ai-company
#
set -e

# スクリプト自身のディレクトリを取得
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WATCHDOG_PATH="${SCRIPT_DIR}/watchdog.py"

# watchdog.py の存在確認
if [ ! -f "$WATCHDOG_PATH" ]; then
    echo "ERROR: watchdog.py not found at ${WATCHDOG_PATH}"
    exit 1
fi

# 環境変数の読み取り（デフォルト値あり）
HEARTBEAT_PATH="${HEARTBEAT_PATH:-/opt/apps/ai-company/companies/default/state/heartbeat.json}"
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"
CONTAINER_NAME="${CONTAINER_NAME:-ai-company}"

# cron エントリを構築
CRON_ENTRY="*/5 * * * * HEARTBEAT_PATH=${HEARTBEAT_PATH} SLACK_WEBHOOK_URL=${SLACK_WEBHOOK_URL} CONTAINER_NAME=${CONTAINER_NAME} /usr/bin/python3 ${WATCHDOG_PATH} >> /var/log/watchdog.log 2>&1"

# 既存の watchdog cron エントリを除去し、新しいエントリを追加
(crontab -l 2>/dev/null | grep -v "watchdog.py" || true; echo "$CRON_ENTRY") | crontab -

echo "✅ Watchdog cron ジョブをインストールしました（5分間隔）"
echo "   スクリプト: ${WATCHDOG_PATH}"
echo "   Heartbeat:  ${HEARTBEAT_PATH}"
echo "   コンテナ:   ${CONTAINER_NAME}"
echo "   ログ:       /var/log/watchdog.log"
