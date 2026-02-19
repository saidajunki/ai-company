# 擬音ショートストーリー自動生成CLI

日本語の「擬音」「擬態語」をもとに、GPT系大言語モデル(OpenAI)で面白いショートストーリー（小話）を自動生成するコマンドラインツールです。

## 特徴

- 擬音/擬態語データセットを活用（`data/gion_list.txt`）
- ランダム抽出/指定で複数の擬音をストーリーに必ず盛り込む
- GPT-3.5, GPT-4（OpenAI API）に対応
- シェルから `./gion-story` 一発で実行OK
- サンプル出力をすぐ確認出来る

## インストール・準備

```sh
# 仮想環境推奨
python3 -m venv .venv
source .venv/bin/activate
pip install openai
export OPENAI_API_KEY=sk-... # ご自身のAPIキーをここに
```

## 使い方

```sh
cd gion-story-gen
./gion-story
# または直接:
python gion_story_gen.py --num 3
```

オプション例:
- `--num 7` : 使う擬音語の数を7個に
- `--gion ぺこぺこ --gion もぐもぐ` : 指定した擬音語のみ利用
- `--debug` : GPTプロンプトを表示
- `--model gpt-4` : GPT-4も利用可

## サンプル出力

> 例（`--num 3` の一例）  
>  
> 「ぺこぺこ」のお腹をかかえて、たろうは「もぐもぐ」とパンを食べ始めた。すると口元には「きらきら」とジャムが光っていた。今日も平和な朝だった。

## ファイル構成

- `gion_story_gen.py` ... Python本体（中核ロジック）
- `gion-story` ... シェルラッパー
- `data/gion_list.txt` ... 擬音・擬態語リスト
- `LICENSE`, `.gitignore`, `README.md` ... 各種管理ファイル

## ライセンス

MITライセンス ― ご自由にご利用・改変OK！

