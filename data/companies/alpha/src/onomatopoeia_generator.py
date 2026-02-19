import random

KANA_HEADS = ["ド", "バ", "ガ", "ズ", "ボ", "パ", "サ", "キ", "タ", "ニ"]
KANA_BODYS = ["ン", "キ", "カ", "シュ", "バ", "リ", "ン", "ッ", "チ"]
REPEAT_NUM = [1, 2, 3]

def generate_onomatopoeia():
    head = random.choice(KANA_HEADS)
    body = "".join([random.choice(KANA_BODYS) for _ in range(random.choice(REPEAT_NUM))])
    return head + body

if __name__ == "__main__":
    for _ in range(5):
        print(generate_onomatopoeia())
