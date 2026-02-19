#!/usr/bin/env python3
import os
import json
import requests

# 必須: Slack Bot Token (xoxb-...)
SLACK_BOT_TOKEN = os.environ.get("FUNNY_NEWS_SLACK_BOT_TOKEN", "")
SLACK_CHANNEL_ID = os.environ.get("FUNNY_NEWS_CHANNEL_ID", "")
NEWS_LOG_PATH = "./src/funny_news/distribute_log.ndjson"
OUTPUT_PATH = "./src/funny_news/feedback_log.ndjson"

def get_slack_reactions_and_replies(ts, channel):
    """Slackの特定メッセージのreaction数と返信数を取得"""
    headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
    react_url = "https://slack.com/api/reactions.get"
    resp = requests.get(react_url, params={"channel": channel, "timestamp": ts}, headers=headers)
    reactions = []
    if resp.ok:
        resj = resp.json()
        if resj.get("ok") and resj.get("message"):
            reactions = resj["message"].get("reactions", [])
    total_reactions = sum([r.get("count",0) for r in reactions])
    # スレッド返信数取得
    replies_url = "https://slack.com/api/conversations.replies"
    r2 = requests.get(replies_url, params={"channel": channel, "ts": ts}, headers=headers)
    n_replies = 0
    if r2.ok:
        retj = r2.json()
        if retj.get("ok") and retj.get("messages"):
            msgs = retj["messages"]
            if len(msgs) > 0:
                n_replies = msgs[0].get("reply_count", 0)
    return total_reactions, n_replies

def calc_funny_score(n_reactions, n_replies):
    """面白さスコア 定義: reaction数 + 2*返信数（初期案）"""
    return n_reactions + 2 * n_replies

def main():
    if not SLACK_BOT_TOKEN or not SLACK_CHANNEL_ID:
        print("Slack BOT TOKENやChannel ID未設定")
        return
    result_lines = []
    with open(NEWS_LOG_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ts = d.get("slack_ts", "")
            if not ts: continue
            n_react, n_reply = get_slack_reactions_and_replies(ts, SLACK_CHANNEL_ID)
            funny_score = calc_funny_score(n_react, n_reply)
            record = {
                "ts": ts,
                "news": d.get("news"),
                "n_reactions": n_react,
                "n_replies": n_reply,
                "funny_score": funny_score,
                "timestamp": d.get("timestamp"),
            }
            print(f"ニュース: {d.get('news')}\n👀Reaction: {n_react}, 💬返信: {n_reply}, 🎯面白スコア: {funny_score}")
            result_lines.append(record)
    # 保存
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in result_lines:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
