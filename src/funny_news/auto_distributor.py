#!/usr/bin/env python3
import json
import os
import random
import sys
import time
import requests
from datetime import datetime

# 設定
SLACK_WEBHOOK_URL = os.environ.get("FUNNY_NEWS_SLACK_WEBHOOK", "")
NEWS_LOG_PATH = "./src/funny_news/distribute_log.ndjson"
JOKES = [
    "【新作発表】OpenAIの新AI、会議で寝落ち検知！爆音で叩き起こします。",
    "【速報】AIが自分のジョークにウケてしまい、エラーでフリーズ！",
    "【話題】Stable Diffusion、ウソみたいな犬画像を自動生成！飼いたくはない。",
    "【衝撃】Copilotが脚本を書いたコント、意外とウケる（社内限定）。"
]

def make_news():
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    news = random.choice(JOKES)
    return f"[{now}] 面白AIニュース:\n{news}"

def send_slack(text):
    """Slackへ投稿し、投稿成功時はts等も返す"""
    if not SLACK_WEBHOOK_URL:
        print("SLACK_WEBHOOK_URL未設定")
        return False, ""
    payload = {"text": text}
    resp = requests.post(SLACK_WEBHOOK_URL, data=json.dumps(payload), headers={'Content-Type': 'application/json'})
    try:
        result = resp.json()
    except Exception:
        result = {}
    # Slack Incoming Webhookはts返さない場合もあり（API連携用に拡張余地）
    if resp.status_code == 200:
        return True, result.get("ts", "")
    return False, ""

def log_distribution(news_text, sl_ts=""):
    dct = {"timestamp": datetime.now().isoformat(), "news": news_text, "slack_ts": sl_ts}
    with open(NEWS_LOG_PATH, "a") as f:
        f.write(json.dumps(dct, ensure_ascii=False) + "\n")

def main():
    news_text = make_news()
    ok, sl_ts = send_slack(news_text)
    log_distribution(news_text, sl_ts)
    print(f"配信 {'成功' if ok else '失敗'}: {news_text} (Slack ts={sl_ts})")

if __name__ == "__main__":
    main()
