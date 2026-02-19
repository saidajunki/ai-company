import random
from gtts import gTTS

def load_words():
    with open("giongo_words.txt", encoding="utf-8") as f:
        return [w.strip() for w in f if w.strip()]

def generate_rap():
    words = load_words()
    rhyme_dict = {}
    for w in words:
        suffix = w[-2:]
        rhyme_dict.setdefault(suffix, []).append(w)

    candidate_rhymes = [k for k, v in rhyme_dict.items() if len(v) >= 3]
    if candidate_rhymes:
        rhyme_suffix = random.choice(candidate_rhymes)
        rhyme_words = rhyme_dict[rhyme_suffix]
    else:
        rhyme_words = random.sample(words, 9)
    line1 = "　".join(random.sample(rhyme_words, 3))
    line2 = "　".join(random.sample(rhyme_words, 3))
    line3 = "　".join(random.sample(rhyme_words, 3))
    lyric = f"{line1}\n{line2}\n{line3}\n響け韻！AIラップタイム"
    return lyric

if __name__ == "__main__":
    print(generate_rap())
