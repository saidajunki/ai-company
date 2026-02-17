#!/bin/bash
# =============================================================================
# AI Company — Docker → VPSホスト直接実行 移行スクリプト
#
# 処理フロー:
#   1. 前提条件チェック（Python, Docker volume）
#   2. Docker named volumeのデータをホストにコピー
#   3. Python仮想環境のセットアップ
#   4. .envファイルの準備
#   5. systemdサービスの登録・起動
#   6. ヘルスチェック
#   7. 成功/失敗時の案内表示
#
# 使い方: sudo bash migrate-to-host.sh
# =============================================================================

set -euo pipefail

APP_DIR="/opt/apps/ai-company"
DATA_DIR="${APP_DIR}/data"
VENV_DIR="${APP_DIR}/.venv"
ENV_FILE="${APP_DIR}/.env"
ENV_TEMPLATE="${APP_DIR}/.env.template"
SERVICE_FILE="${APP_DIR}/ai-company.service"
SYSTEMD_TARGET="/etc/systemd/system/ai-company.service"
DOCKER_VOLUME="ai-company_ai-company-data"
BACKUP_DIR="${APP_DIR}/data-backup-$(date +%Y%m%d-%H%M%S)"
CONTAINER_NAME="ai-company"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()   { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ---------------------------------------------------------------------------
# 1. 前提条件チェック
# ---------------------------------------------------------------------------
log "=== AI Company 移行スクリプト ==="
log ""

# root チェック
if [ "$(id -u)" -ne 0 ]; then
    error "root権限が必要です。sudo bash migrate-to-host.sh で実行してください。"
    exit 1
fi

# Python 3.11+ チェック
if ! command -v python3 &>/dev/null; then
    error "python3 が見つかりません。Python 3.11以上をインストールしてください。"
    error "  apt update && apt install -y python3 python3-pip python3-venv"
    exit 1
fi

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
    error "Python ${PY_VERSION} が検出されました。3.11以上が必要です。"
    exit 1
fi
log "Python ${PY_VERSION} ✓"

# ソースディレクトリチェック
if [ ! -f "${APP_DIR}/pyproject.toml" ]; then
    error "${APP_DIR}/pyproject.toml が見つかりません。リポジトリが正しくクローンされていますか？"
    exit 1
fi
log "ソースディレクトリ ${APP_DIR} ✓"

# ---------------------------------------------------------------------------
# 2. Docker named volumeのデータをホストにコピー
# ---------------------------------------------------------------------------
log ""
log "--- Step 2: データ移行 ---"

# Docker volumeの存在チェック
VOLUME_EXISTS=$(docker volume ls -q | grep -c "^${DOCKER_VOLUME}$" || true)

if [ "$VOLUME_EXISTS" -eq 0 ]; then
    warn "Docker volume '${DOCKER_VOLUME}' が見つかりません。"
    if [ -d "$DATA_DIR" ] && [ "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
        log "既存のデータディレクトリ ${DATA_DIR} を使用します。"
    else
        warn "データディレクトリが空です。新規インストールとして続行します。"
        mkdir -p "$DATA_DIR"
    fi
else
    # 既存データのバックアップ
    if [ -d "$DATA_DIR" ] && [ "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
        log "既存データをバックアップ: ${BACKUP_DIR}"
        cp -a "$DATA_DIR" "$BACKUP_DIR"
    fi

    # 旧コンテナの停止
    if docker ps -q -f "name=${CONTAINER_NAME}" | grep -q .; then
        log "旧コンテナ '${CONTAINER_NAME}' を停止中..."
        docker stop "${CONTAINER_NAME}" || true
    fi

    # Docker volumeからデータをコピー
    log "Docker volume '${DOCKER_VOLUME}' からデータをコピー中..."
    mkdir -p "$DATA_DIR"
    # 一時コンテナを使ってvolumeの中身をコピー
    docker run --rm \
        -v "${DOCKER_VOLUME}:/source:ro" \
        -v "${DATA_DIR}:/dest" \
        ubuntu:latest \
        bash -c "cp -a /source/. /dest/"
    log "データコピー完了 ✓"
fi

# ---------------------------------------------------------------------------
# 3. Python仮想環境のセットアップ
# ---------------------------------------------------------------------------
log ""
log "--- Step 3: Python仮想環境セットアップ ---"

if [ -d "$VENV_DIR" ]; then
    log "既存の仮想環境を検出。依存関係を更新します。"
else
    log "仮想環境を作成中: ${VENV_DIR}"
    python3 -m venv "$VENV_DIR"
fi

log "依存関係をインストール中..."
"${VENV_DIR}/bin/pip" install --upgrade pip -q
"${VENV_DIR}/bin/pip" install "." -q
log "仮想環境セットアップ完了 ✓"

# ---------------------------------------------------------------------------
# 4. .envファイルの準備
# ---------------------------------------------------------------------------
log ""
log "--- Step 4: 環境変数ファイル ---"

if [ -f "$ENV_FILE" ]; then
    log ".env ファイルが既に存在します。既存の設定を維持します。"
else
    if [ -f "$ENV_TEMPLATE" ]; then
        log ".env.template から .env を作成します。"
        cp "$ENV_TEMPLATE" "$ENV_FILE"
        warn "⚠ .env ファイルの秘密情報（SLACK_BOT_TOKEN, OPENROUTER_API_KEY等）を設定してください。"
    else
        error ".env.template が見つかりません。手動で .env を作成してください。"
        exit 1
    fi
fi

chmod 600 "$ENV_FILE"
log ".env 権限設定 (600) ✓"

# ---------------------------------------------------------------------------
# 5. systemdサービスの登録・起動
# ---------------------------------------------------------------------------
log ""
log "--- Step 5: systemdサービス登録 ---"

if [ ! -f "$SERVICE_FILE" ]; then
    error "サービスファイル ${SERVICE_FILE} が見つかりません。"
    exit 1
fi

cp "$SERVICE_FILE" "$SYSTEMD_TARGET"
systemctl daemon-reload
systemctl enable ai-company
log "systemdサービス登録完了 ✓"

log "サービスを起動中..."
systemctl start ai-company

# ---------------------------------------------------------------------------
# 6. ヘルスチェック
# ---------------------------------------------------------------------------
log ""
log "--- Step 6: ヘルスチェック ---"
sleep 5

if systemctl is-active --quiet ai-company; then
    log "✅ ai-company サービスは正常に起動しています"
    log ""
    log "=== 移行成功 ==="
    log ""
    log "確認コマンド:"
    log "  systemctl status ai-company"
    log "  journalctl -u ai-company -f"
    log ""
    log "旧Dockerリソースの削除（任意）:"
    log "  docker rm ${CONTAINER_NAME}"
    log "  docker volume rm ${DOCKER_VOLUME}"
    log "  docker rmi ai-company:latest"
else
    error "❌ ai-company サービスの起動に失敗しました"
    error ""
    error "ログを確認:"
    error "  journalctl -u ai-company --no-pager -n 50"
    error ""
    error "=== ロールバック手順 ==="
    error "  systemctl stop ai-company"
    error "  systemctl disable ai-company"
    error "  rm ${SYSTEMD_TARGET}"
    error "  systemctl daemon-reload"
    if [ "$VOLUME_EXISTS" -gt 0 ]; then
        error "  docker start ${CONTAINER_NAME}"
    fi
    if [ -d "$BACKUP_DIR" ]; then
        error "  rm -rf ${DATA_DIR} && mv ${BACKUP_DIR} ${DATA_DIR}"
    fi
    exit 1
fi
