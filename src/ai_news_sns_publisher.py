import os
import requests

TWITTER_API_URL = "https://api.twitter.com/2/tweets"
TWITTER_MEDIA_UPLOAD_URL = "https://upload.twitter.com/1.1/media/upload.json"

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

def upload_image_to_twitter(image_path: str):
    bearer_token = os.environ.get('TWITTER_BEARER_TOKEN')
    if not bearer_token:
        raise RuntimeError("Twitter Bearer Token not set in environment variable.")

    headers = {"Authorization": f"Bearer {bearer_token}"}
    files = {"media": open(image_path, "rb")}
    resp = requests.post(TWITTER_MEDIA_UPLOAD_URL, headers=headers, files=files)
    try:
        resp.raise_for_status()
        media_id = resp.json().get("media_id_string")
        return media_id, None
    except Exception as e:
        return None, str(e) + ": " + resp.text

def post_image_with_text(image_path: str, text: str):
    media_id, err = upload_image_to_twitter(image_path)
    if err or not media_id:
        return None, f"media upload failed: {err}"
    bearer_token = os.environ.get('TWITTER_BEARER_TOKEN')
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json"
    }
    data = {"text": text, "media": {"media_ids": [media_id]}}
    resp = requests.post(TWITTER_API_URL, headers=headers, json=data)
    try:
        resp.raise_for_status()
        return resp.json(), None
    except Exception as e:
        return None, str(e) + ": " + resp.text

if __name__ == "__main__":
    # 実行例:
    import sys
    text = sys.argv[1] if len(sys.argv) > 1 else "[テスト]AIニュース自動投稿+画像投稿"
    image_path = sys.argv[2] if len(sys.argv) > 2 else "ascii4koma_sample.png"
    DUMMY = os.environ.get("DUMMY_POST", "0") == "1"
    # テキストのみ投稿
    if DUMMY:
        print(f"DUMMY MODE: text={text} image={image_path}")
    else:
        if os.path.exists(image_path):
            res, err = post_image_with_text(image_path, text)
        else:
            res, err = post_to_twitter(text)
        print("RESULT:", res if res else err)

if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else "ascii4koma_sample.png"
    txt = sys.argv[2] if len(sys.argv) > 2 else "ASCIIアート4コマ漫画自動投稿テスト"
    print(f"画像付き投稿テスト: {img} - {txt}")
    try:
        result, error = post_image_with_text(img, txt)
        if error:
            print("SNS投稿失敗:", error)
        else:
            print("SNS投稿成功:", result)
    except Exception as e:
        print("投稿例外:", e)
