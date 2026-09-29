
"""Kiểm tra giới hạn 8 ảnh/request. Dùng: python3 test-images.py [base_url]"""
import base64, json, struct, sys, urllib.error, urllib.request, zlib

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://vllm-g4-agg:8000"
MODEL = "gemma4-12b"

COLORS = [
    ("red", (255, 0, 0)), ("green", (0, 255, 0)), ("blue", (0, 0, 255)),
    ("yellow", (255, 255, 0)), ("magenta", (255, 0, 255)), ("cyan", (0, 255, 255)),
    ("white", (255, 255, 255)), ("orange", (255, 128, 0)),
    ("purple", (128, 0, 255)), ("pink", (255, 128, 192)),
]


BG = (128, 128, 128)


def png(rgb, size=256):
    """PNG size x size: hinh tron mau rgb tren nen xam, khong dung thu vien ngoai."""
    c, r2 = size / 2, (size * 0.31) ** 2
    raw = b"".join(
        b"\x00" + b"".join(bytes(rgb if (x - c) ** 2 + (y - c) ** 2 <= r2 else BG)
                           for x in range(size))
        for y in range(size))

    def chunk(tag, data):
        body = tag + data
        return (struct.pack(">I", len(data)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


def ask(n):
    content = [
        {"type": "image_url",
         "image_url": {"url": "data:image/png;base64,"
                              + base64.b64encode(png(rgb)).decode()}}
        for _, rgb in COLORS[:n]
    ]
    content.append({
        "type": "text",
        "text": f"Each of the {n} images above shows one colored circle on a "
                f"gray background. List the circle colors in order, "
                f"comma-separated. Answer with color names only.",
    })
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 60, "temperature": 0,
    }).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.load(r)
        return "200  " + d["choices"][0]["message"]["content"].strip().replace("\n", " ")[:150]
    except urllib.error.HTTPError as e:
        return f"{e.code}  " + e.read().decode()[:220]


if __name__ == "__main__":
    expected = " | ".join(name for name, _ in COLORS[:8])
    print(f"mong doi o 8 anh: {expected}\n")
    for n in (8, 9, 10):
        print(f"[{n:>2} anh] {ask(n)}")
