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
# SLACK_WEBHOOK_URL : Slack Incoming Webhook の URL
#   例: https://hooks.slack.com/services/YOUR/WEBHOOK/URL
#
# ===== オプション環境変数 =====
#
# HEARTBEAT_PATH : heartbeat.json のフルパス（未設定なら docker exec 経由で読む）
#
# SYSTEMD_SERVICE_NAME : systemd サービス名（設定すると systemctl restart で復旧する）
#   例: ai-company
#
# CONTAINER_NAME : Docker コンテナ名（デフォルト: ai-company）
#   例: ai-company
#
# COMPANY_ID : 会社ID（デフォルト: alpha）
#
# THRESHOLD_MINUTES : stale 判定閾値（デフォルト: 20）
#
# BASE_DIR_IN_CONTAINER : コンテナ内の BASE_DIR（デフォルト: /app/data）
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
HEARTBEAT_PATH="${HEARTBEAT_PATH-}"
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL-}"
CONTAINER_NAME="${CONTAINER_NAME:-ai-company}"
SYSTEMD_SERVICE_NAME="${SYSTEMD_SERVICE_NAME-}"
COMPANY_ID="${COMPANY_ID:-alpha}"
THRESHOLD_MINUTES="${THRESHOLD_MINUTES:-20}"
BASE_DIR_IN_CONTAINER="${BASE_DIR_IN_CONTAINER:-/app/data}"

# systemd mode: default to host heartbeat path if not provided
if [ -n "${SYSTEMD_SERVICE_NAME}" ] && [ -z "${HEARTBEAT_PATH}" ]; then
  HEARTBEAT_PATH="/opt/apps/ai-company/data/companies/${COMPANY_ID}/state/heartbeat.json"
fi

# cron エントリを構築
CRON_ENTRY="*/5 * * * * HEARTBEAT_PATH=${HEARTBEAT_PATH} SLACK_WEBHOOK_URL=${SLACK_WEBHOOK_URL} CONTAINER_NAME=${CONTAINER_NAME} SYSTEMD_SERVICE_NAME=${SYSTEMD_SERVICE_NAME} COMPANY_ID=${COMPANY_ID} THRESHOLD_MINUTES=${THRESHOLD_MINUTES} BASE_DIR_IN_CONTAINER=${BASE_DIR_IN_CONTAINER} /usr/bin/python3 ${WATCHDOG_PATH} >> /var/log/watchdog.log 2>&1"

# 既存の watchdog cron エントリを除去し、新しいエントリを追加
(crontab -l 2>/dev/null | grep -v "watchdog.py" || true; echo "$CRON_ENTRY") | crontab -

echo "✅ Watchdog cron ジョブをインストールしました（5分間隔）"
echo "   スクリプト: ${WATCHDOG_PATH}"
if [ -n "${HEARTBEAT_PATH}" ]; then
  echo "   Heartbeat:  ${HEARTBEAT_PATH}（ホストパス）"
else
  echo "   Heartbeat:  docker exec 経由（host path 未設定）"
fi
echo "   コンテナ:   ${CONTAINER_NAME}"
echo "   会社ID:     ${COMPANY_ID}"
echo "   閾値:       ${THRESHOLD_MINUTES}分"
echo "   ログ:       /var/log/watchdog.log"
