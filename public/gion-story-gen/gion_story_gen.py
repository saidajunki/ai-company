#!/usr/bin/env python3
import os
import sys
import random
import argparse

try:
    import openai
except ImportError:
    print("openaiライブラリが必要です。pip install openai でインストールしてください", file=sys.stderr)
    sys.exit(1)

def load_gion_list(filepath):
    with open(filepath, encoding="utf-8") as f:
        gions = [line.strip() for line in f if line.strip()]
    return gions

def pick_gions(gions, n):
    return random.sample(gions, k=min(n, len(gions)))

def create_prompt(selected_gions):
    # 指示文
    inp = "、".join(selected_gions)
    return (
        f"以下の日本語の擬音・擬態語を必ず使って、短くて面白いショートストーリー（小話）を日本語で1つ書いてください。\n"
        f"【使う単語】{inp}\n\n"
        f"- すべての擬音語は目立つ形で自然に文中に埋め込んでください。\n"
        f"- シニカルやギャグや日常コメディなどパターンはお任せ。\n"
        f"- 250文字以内。\n"
        f"出力はストーリー本文のみ。"
    )

def main():
    parser = argparse.ArgumentParser(description="擬音・擬態語を活用したショートストーリー自動生成（GPT利用）")
    parser.add_argument("--num", type=int, default=5, help="使う擬音語の数（デフォルト5個）")
    parser.add_argument("--gion", action="append", help="特定の擬音語を直接指定。複数回使用可")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAIモデル名 (例: gpt-4)")
    parser.add_argument("--debug", action='store_true', help="プロンプト内容を表示する")
    args = parser.parse_args()

    gion_file = os.path.join(os.path.dirname(__file__), "data", "gion_list.txt")
    gion_list = load_gion_list(gion_file)

    # 擬音のリスト
    if args.gion:
        selected_gions = args.gion
    else:
        selected_gions = pick_gions(gion_list, args.num)

    prompt = create_prompt(selected_gions)
    if args.debug:
        print(f"---PROMPT---\n{prompt}\n--------------\n", file=sys.stderr)

    openai.api_key = os.environ.get("OPENAI_API_KEY")
    if not openai.api_key:
        print("OPENAI_API_KEY 環境変数が未設定です", file=sys.stderr)
        sys.exit(1)

    # OpenAIチャットAPI呼び出し
    try:
        rsp = openai.ChatCompletion.create(
            model=args.model,
            messages=[
                {"role": "system", "content": "あなたはユーモア小説家です。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.95
        )
        story = rsp.choices[0].message['content'].strip()
        print(story)
    except Exception as e:
        print(f"OpenAI APIエラー: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
