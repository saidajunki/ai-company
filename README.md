# ai-company

AIエージェント中心で運用する「会社（組織）」の構想・要件・設計・進め方（将来的な実装コードも含む）を、このリポジトリに集約する。

## 目的

- Slack上で意思決定/相談/報告を回しながら、VPS上でAIが自律的に作業を進める
- 人間（Creator）は原則として **契約・支払い・アカウント作成などの手続き** と **承認** のみを担当する
- 成果物は **原則公開**（確認を簡単にするため）
  - ただし `APIキー/トークン/支払い情報/個人情報` は例外として公開しない（課金事故の防止）

## ドキュメント

- `ai-company/docs/00_vision.md`: 目指す姿（何を実現したいか）
- `ai-company/docs/01_requirements.md`: 具体要件（Slack/コスト/人間介入/公開方針など）
- `ai-company/docs/02_design.md`: 設計（構成要素、記憶、台帳、情報収集、承認フロー）
- `ai-company/docs/03_roadmap.md`: 進め方（タスク/マイルストーン）
- `ai-company/docs/decision-log.md`: 意思決定ログ（いつ・何を・なぜ・見直し条件）
-e 
# ASCII4コマ漫画 変換・SNS自動投稿

python src/ascii4koma_to_svg.py # SVG/PNG生成
python src/ai_news_sns_publisher.py ascii4koma_sample.png '投稿例' # SNS投稿
(要 requests/cairosvg/Token)

