#!/bin/bash
# =============================================================================
# AI Company — Python仮想環境セットアップスクリプト
#
# 使い方: bash setup-venv.sh
# =============================================================================

set -euo pipefail

APP_DIR="/opt/apps/ai-company"
VENV_DIR="${APP_DIR}/.venv"

# Python 3.11+ チェック
if ! command -v python3 &>/dev/null; then
    echo "❌ python3 が見つかりません。"
    echo "   apt update && apt install -y python3 python3-pip python3-venv"
    exit 1
fi

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
    echo "❌ Python ${PY_VERSION} が検出されました。3.11以上が必要です。"
    exit 1
fi
echo "✓ Python ${PY_VERSION}"

cd "$APP_DIR"

if [ -d "$VENV_DIR" ]; then
    echo "既存の仮想環境を検出。依存関係を更新します。"
else
    echo "仮想環境を作成中..."
    python3 -m venv "$VENV_DIR"
fi

echo "依存関係をインストール中..."
"${VENV_DIR}/bin/pip" install --upgrade pip -q
"${VENV_DIR}/bin/pip" install ".[dev]" -q

echo "✅ セットアップ完了: ${VENV_DIR}"
