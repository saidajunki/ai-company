import sys
import json
import datetime
from collections import defaultdict

def load_feedback(file_in):
    with open(file_in, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except Exception:
                continue

def summarize(feedbacks):
    # 日毎カウント
    summary = defaultdict(lambda: {"posts": 0, "comments": 0, "reactions": 0, "top_post": None})
    for fb in feedbacks:
        t = datetime.datetime.fromtimestamp(float(fb['ts']))
        day = t.strftime("%Y-%m-%d")
        summary[day]['posts'] += 1
        summary[day]['comments'] += fb.get('reply_count', 0)
        r_cnt = sum([r.get('count', 0) for r in fb.get('reactions', [])])
        summary[day]['reactions'] += r_cnt
        # 反響多い投稿
        score = r_cnt + fb.get('reply_count', 0)
        if not summary[day]['top_post'] or summary[day]['top_post']['score'] < score:
            summary[day]['top_post'] = {
                'score': score,
                'text': fb.get('text', '')[:50],
                'user': fb.get('user'),
                'ts': fb.get('ts')
            }
    return summary

def print_report(summary):
    for day in sorted(summary.keys()):
        print(f"[{day}] 投稿数: {summary[day]['posts']} コメント数: {summary[day]['comments']} リアクション数: {summary[day]['reactions']}")
        post = summary[day]['top_post']
        if post:
            print(f"  反響最大: {post['score']}件 : ユーザ:{post['user']} '{post['text']}'")

if __name__ == "__main__":
    in_file = sys.argv[1] if len(sys.argv) > 1 else "feedback.jsonl"
    feedbacks = list(load_feedback(in_file))
    summary = summarize(feedbacks)
    print_report(summary)
