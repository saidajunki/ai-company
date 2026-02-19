import os
import requests

TWITTER_API_URL = "https://api.twitter.com/2/tweets/{}?tweet.fields=public_metrics"

def get_twitter_post_metrics(tweet_id: str):
    bearer_token = os.environ.get('TWITTER_BEARER_TOKEN')
    if not bearer_token:
        raise RuntimeError("Twitter Bearer Token not set in environment variable.")

    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json"
    }
    resp = requests.get(TWITTER_API_URL.format(tweet_id), headers=headers)
    try:
        resp.raise_for_status()
        js = resp.json()
        metrics = js["data"]["public_metrics"]
        return metrics, None
    except Exception as e:
        return None, str(e) + ": " + resp.text

if __name__ == "__main__":
    # テスト用：Tweet IDをCLI引数で受取
    import sys
    tweet_id = sys.argv[1] if len(sys.argv) > 1 else "0000000000"
    if os.environ.get("DUMMY_METRICS", "0") == "1":
        dummy = {"retweet_count": 1, "reply_count": 2, "like_count": 3, "quote_count": 0}
        print(f"DUMMY METRICS: {tweet_id} => {dummy}")
    else:
        metrics, err = get_twitter_post_metrics(tweet_id)
        print("RESULT:", metrics if metrics else err)
