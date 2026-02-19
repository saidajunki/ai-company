import os
import subprocess

def test_dry_run():
    result = subprocess.run(
        ["python3", "src/sns_metrics_collector.py", "12345678"],
        env={**os.environ, "DUMMY_METRICS": "1"},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    assert "DUMMY METRICS" in result.stdout, "Dry-run should not call Twitter API"

def test_missing_token():
    env_no_token = {**os.environ, "DUMMY_METRICS": "0"}
    if 'TWITTER_BEARER_TOKEN' in env_no_token:
        del env_no_token['TWITTER_BEARER_TOKEN']
    result = subprocess.run(
        ["python3", "src/sns_metrics_collector.py", "87654321"],
        env=env_no_token,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    assert "Bearer Token not set" in result.stderr or "Bearer Token not set" in result.stdout

if __name__ == "__main__":
    test_dry_run()
    test_missing_token()
    print("sns_metrics_collector.py テスト PASS")
