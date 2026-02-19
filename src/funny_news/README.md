# 面白AIニュース解説 自動配信スクリプト

## 概要
- 週3回（cron/手動）Slackへ面白AIニュース（ランダム小ネタ）を配信
- 配信ログ（ndjson）を自動蓄積
- 今後、反響データ取り込み・面白さスコア自動算出へ拡張予定

## 環境変数
- FUNNY_NEWS_SLACK_WEBHOOK : Slack Incoming WebhookのURL

## 実行方法
```sh
cd /opt/apps/ai-company
FUNNY_NEWS_SLACK_WEBHOOK=https://hooks.slack.com/services/... ./src/funny_news/auto_distributor.py
```

## 定期実行例（週3回・月水金朝8:00）
```cron
0 8 * * 1,3,5 cd /opt/apps/ai-company && FUNNY_NEWS_SLACK_WEBHOOK='...' ./src/funny_news/auto_distributor.py >> /tmp/funny_news.log 2>&1
```

## ログファイル
- ./src/funny_news/distribute_log.ndjson

1行1配信, 例:
{"timestamp": "2026-02-20T08:51:00", "news": "[2026-02-20 08:51] 面白AIニュース: ..."}

## 今後・TODO
- Slack reaction数・スレッド返信数をAPIで取得し反響を記録
- 反響から面白さスコア生成
```
