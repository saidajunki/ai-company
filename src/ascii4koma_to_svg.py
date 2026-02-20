from typing import List

def ascii4koma_to_svg(panels: List[str], width=400, height=600, margin=10, font_size=14) -> str:
    """
    4コマ漫画用のASCIIテキストリストをSVG画像に変換
    panels: 各コマごとのASCIIテキスト(list of str, 長さ4想定)
    """
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<style>text { font-family: monospace; white-space: pre; }</style>'
    ]
    koma_h = (height - (margin * 5)) // 4
    for i, text in enumerate(panels):
        y = margin + i * (koma_h + margin)
        # コマ枠
        svg_parts.append(
            f'<rect x="{margin}" y="{y}" width="{width - 2*margin}" height="{koma_h}" fill="#fff" stroke="#222"/>'
        )
        # テキスト（行ごと分割表示、等幅＆行送り）
        lines = text.split('\n')
        for j, line in enumerate(lines):
            ty = y + font_size + j * (font_size + 2)
            svg_parts.append(
                f'<text x="{margin + 8}" y="{ty}" font-size="{font_size}">{line}</text>'
            )
    svg_parts.append('</svg>')
    return '\n'.join(svg_parts)

if __name__ == "__main__":
    ascii_4koma = [
        "＿人人人人人＿\n＞  １コマ目 ＜\n￣Y^Y^Y^Y￣",
        " (^o^)＜うぇーい\nASCIIのまま！",
        " (´・ω・｀)\n２コマ寝落ち",
        "完    END\n感謝！"
    ]
    svg = ascii4koma_to_svg(ascii_4koma)
    with open("ascii4koma_sample.svg", "w") as f:
        f.write(svg)
    print("SVGファイル 'ascii4koma_sample.svg' 出力済")

def svg_to_png(svg_path: str, out_png_path: str):
    import cairosvg
    with open(svg_path, "r") as f:
        svg_data = f.read()
    cairosvg.svg2png(bytestring=svg_data.encode('utf-8'), write_to=out_png_path)

if __name__ == "__main__":
    ascii_4koma = [
        "＿人人人人人＿\n＞  １コマ目 ＜\n￣Y^Y^Y^Y￣",
        " (^o^)＜うぇーい\nASCIIのまま！",
        " (´・ω・｀)\n２コマ寝落ち",
        "完    END\n感謝！"
    ]
    svg_file = "ascii4koma_sample.svg"
    png_file = "ascii4koma_sample.png"
    svg = ascii4koma_to_svg(ascii_4koma)
    with open(svg_file, "w") as f:
        f.write(svg)
    print(f"SVGファイル '{svg_file}' 出力済")
    # SVG→PNG変換
    try:
        svg_to_png(svg_file, png_file)
        print(f"PNGファイル '{png_file}' への変換も成功")
    except Exception as e:
        print(f"SVG→PNG変換エラー: {e}")
