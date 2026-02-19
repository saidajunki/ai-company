import os
import requests

TWITTER_API_URL = "https://api.twitter.com/2/tweets"

def post_to_twitter(text: str):
    bearer_token = os.environ.get('TWITTER_BEARER_TOKEN')
    if not bearer_token:
        raise RuntimeError("Twitter Bearer Token not set in environment variable.")

    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json"
    }
    data = {"text": text}
    resp = requests.post(TWITTER_API_URL, headers=headers, json=data)
    try:
        resp.raise_for_status()
        return resp.json(), None
    except Exception as e:
        return None, str(e) + ": " + resp.text

if __name__ == "__main__":
    # ドライラン動作: 実行時にDUMMY_POSTが"1"なら投稿せず出力だけ
    import sys
    text = sys.argv[1] if len(sys.argv) > 1 else "[テスト]AIニュース自動投稿スクリプト動作確認"
    if os.environ.get("DUMMY_POST", "0") == "1":
        print(f"DUMMY MODE: {text}")
    else:
        res, err = post_to_twitter(text)
        print("RESULT:", res if res else err)
