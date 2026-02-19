import csv
import random
from datetime import datetime
from onomatopoeia_generator import generate_onomatopoeia
from templates import announce_battle, announce_winner, announce_draw

HISTORY_CSV = "battle_history.csv"

def run_battle(player1, player2):
    ono1 = generate_onomatopoeia()
    ono2 = generate_onomatopoeia()
    print(announce_battle(player1, ono1, player2, ono2))
    result = random.choice(["P1", "P2", "DRAW"])
    if result == "P1":
        print(announce_winner(player1, ono1))
        winner = player1
        winning_ono = ono1
    elif result == "P2":
        print(announce_winner(player2, ono2))
        winner = player2
        winning_ono = ono2
    else:
        print(announce_draw())
        winner = "DRAW"
        winning_ono = ""
    # ログ保存
    with open(HISTORY_CSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), player1, ono1, player2, ono2, winner, winning_ono])
    return winner

def show_score():
    score_dict = {}
    try:
        with open(HISTORY_CSV, encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                winner = row[5]
                if winner not in ["", "DRAW"]:
                    score_dict[winner] = score_dict.get(winner, 0) + 1
        print("=== 勝利数ランキング ===")
        for p, s in sorted(score_dict.items(), key=lambda x: -x[1]):
            print(f"{p}: {s}勝")
    except FileNotFoundError:
        print("まだバトル履歴がありません。")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="擬音バトル復活祭 - CLI")
    parser.add_argument("--battle", nargs=2, metavar=("PLAYER1", "PLAYER2"), help="バトル開始")
    parser.add_argument("--score", action="store_true", help="スコア表示")
    args = parser.parse_args()
    if args.battle:
        run_battle(args.battle[0], args.battle[1])
    if args.score:
        show_score()
