# 擬態語・擬音語 AIラップ歌詞自動生成 プロトタイプ
## 概要
- 日本語の擬態語・擬音語から、韻を重視したAIラップ歌詞を自動生成
- 音声合成（gTTS）も組み込める
- Web API＋簡易GUIデモ付き（Flask）

## セットアップ

### 1. 依存パッケージ（venv推奨）
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

### 2. サーバ起動
./venv/bin/python api_server.py

### 3. Webから利用
- ブラウザで http://localhost:5000/ を開く
- 「ラップを生成！」ボタンで即歌詞生成

### 4. API利用例
curl http://localhost:5000/api/generate_rap

## 構成
- generate_rap.py ... 歌詞生成ロジック（関数化、import可）
- api_server.py   ... Flask API+静的GUIサーバ
- static/index.html ... シンプルなWebデモGUI
- giongo_words.txt ... 単語リスト
- requirements.txt ... 必要な依存
