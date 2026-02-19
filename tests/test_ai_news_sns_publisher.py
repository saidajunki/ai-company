import os
import subprocess

def test_dry_run():
    result = subprocess.run(
        ["python3", "src/ai_news_sns_publisher.py", "AIニュースDRYRUN"],
        env={**os.environ, "DUMMY_POST": "1"},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    assert "DUMMY MODE" in result.stdout, "Dry-run should not post to Twitter"

def test_missing_token():
    env_no_token = {**os.environ, "DUMMY_POST": "0"}
    if 'TWITTER_BEARER_TOKEN' in env_no_token:
        del env_no_token['TWITTER_BEARER_TOKEN']
    result = subprocess.run(
        ["python3", "src/ai_news_sns_publisher.py", "AIニュースTOKENFAIL"],
        env=env_no_token,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    assert "Bearer Token not set" in result.stderr or "Bearer Token not set" in result.stdout

if __name__ == "__main__":
    test_dry_run()
    test_missing_token()
    print("ai_news_sns_publisher.py テスト PASS")
