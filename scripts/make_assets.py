"""Generates the illustrations in assets/ (run once: python scripts/make_assets.py)."""
import math
import os
import random

from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")
os.makedirs(OUT, exist_ok=True)
FONT = "/usr/share/fonts/truetype/google-fonts/Poppins-Medium.ttf"
S = 2  # supersampling factor


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def gradient(w, h, c1, c2):
    base = Image.new("RGB", (w, h), c1)
    top = Image.new("RGB", (w, h), c2)
    mask = Image.new("L", (w, h))
    px = mask.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = int(255 * min(1, max(0, (x / w * 0.75 + y / h * 0.25))))
    return Image.composite(top, base, mask)


def drop(draw, cx, cy, r, fill):
    """Teardrop: circle + triangle on top."""
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
    k = r * 1.0
    pts = [(cx, cy - r * 2.1), (cx - r * math.sin(math.radians(60)) , cy - r * 0.5 ),
           (cx + r * math.sin(math.radians(60)), cy - r * 0.5)]
    draw.polygon(pts, fill=fill)


def hero():
    w, h = 1600 * S, 480 * S
    img = gradient(w, h, (9, 52, 66), (14, 140, 148)).convert("RGBA")
    ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    random.seed(7)
    for _ in range(26):  # soft bokeh circles
        r = random.randint(30, 150) * S
        x, y = random.randint(int(w * 0.35), w), random.randint(0, h)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, random.randint(8, 22)))
    # ECG line
    pts, x, y0 = [], 0, int(h * 0.80)
    beat = [(0, 0), (40, 0), (55, -30), (70, 20), (90, -190), (112, 120), (128, 0), (170, 0),
            (190, -40), (215, 0), (260, 0)]
    while x < w:
        for dx, dy in beat:
            pts.append((x + dx * S * 1.4, y0 + dy * S * 1.0))
        x += 260 * S * 1.4
    d.line(pts, fill=(255, 255, 255, 70), width=5 * S, joint="curve")
    img = Image.alpha_composite(img, ov)

    # glowing big drop on the right
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    cx, cy, r = int(w * 0.80), int(h * 0.58), 120 * S
    drop(gd, cx, cy, int(r * 1.35), (255, 255, 255, 60))
    glow = glow.filter(ImageFilter.GaussianBlur(40 * S))
    img = Image.alpha_composite(img, glow)
    dr = ImageDraw.Draw(img)
    drop(dr, cx, cy, r, (239, 83, 80, 255))
    # highlight
    dr.ellipse([cx - r * 0.62, cy - r * 0.35, cx - r * 0.30, cy + r * 0.25], fill=(255, 255, 255, 120))
    # small drops
    drop(dr, int(w * 0.66), int(h * 0.32), 38 * S, (255, 255, 255, 190))
    drop(dr, int(w * 0.93), int(h * 0.30), 28 * S, (255, 255, 255, 150))
    drop(dr, int(w * 0.70), int(h * 0.80), 24 * S, (255, 183, 77, 230))
    # plus signs
    for (px, py, s) in [(0.55, 0.18, 22), (0.97, 0.78, 18), (0.60, 0.62, 14)]:
        x, y, s = int(w * px), int(h * py), s * S
        dr.rectangle([x - s, y - s // 4, x + s, y + s // 4], fill=(255, 255, 255, 130))
        dr.rectangle([x - s // 4, y - s, x + s // 4, y + s], fill=(255, 255, 255, 130))
    img.convert("RGB").resize((w // S, h // S), Image.LANCZOS).save(os.path.join(OUT, "hero.png"), optimize=True)


def broccoli(d, cx, cy, s):
    d.rounded_rectangle([cx - s * 0.12, cy, cx + s * 0.12, cy + s * 0.55], radius=int(s * 0.08), fill=(139, 195, 74))
    for dx, dy, rr in [(-0.28, -0.1, .3), (0.28, -0.1, .3), (0, -0.32, .34), (-0.12, 0.05, .28), (0.14, 0.05, .28)]:
        d.ellipse([cx + dx * s - rr * s, cy + dy * s - rr * s, cx + dx * s + rr * s, cy + dy * s + rr * s],
                  fill=(46, 125, 50))
    for dx, dy in [(-0.2, -0.3), (0.1, -0.4), (0.25, -0.05), (-0.3, 0.0)]:
        d.ellipse([cx + dx * s - 5, cy + dy * s - 5, cx + dx * s + 5, cy + dy * s + 5], fill=(102, 187, 106))


def plate():
    W = 900 * S
    img = Image.new("RGB", (W, W), (240, 249, 248))
    d = ImageDraw.Draw(img)
    c, R = W // 2, int(W * 0.40)
    # shadow
    sh = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    ImageDraw.Draw(sh).ellipse([c - R - 6, c - R + 26, c + R + 6, c + R + 38], fill=(0, 60, 70, 70))
    sh = sh.filter(ImageFilter.GaussianBlur(22 * S))
    img = Image.alpha_composite(img.convert("RGBA"), sh).convert("RGB")
    d = ImageDraw.Draw(img)
    d.ellipse([c - R, c - R, c + R, c + R], fill=(255, 255, 255), outline=(207, 224, 226), width=6 * S)
    r = int(R * 0.82)
    box = [c - r, c - r, c + r, c + r]

    def section(start, end, color, decorate):
        layer = Image.new("RGB", (W, W), color)
        ld = ImageDraw.Draw(layer)
        decorate(ld)
        m = Image.new("L", (W, W), 0)
        ImageDraw.Draw(m).pieslice(box, start, end, fill=255)
        img.paste(layer, (0, 0), m)

    random.seed(3)

    def veg(ld):
        for _ in range(7):
            broccoli(ld, random.randint(c - r + 80 * S, c - 70 * S), random.randint(c - r + 90 * S, c + r - 150 * S), 95 * S)
        for _ in range(8):
            x, y = random.randint(c - r, c - 40 * S), random.randint(c - r, c + r)
            ld.ellipse([x - 26 * S, y - 26 * S, x + 26 * S, y + 26 * S], fill=(229, 57, 53))
            ld.ellipse([x - 14 * S, y - 14 * S, x + 14 * S, y + 14 * S], fill=(239, 108, 100))

    def protein(ld):
        for i in range(2):
            x0, y0 = c + 70 * S + i * 40 * S, c - r + 120 * S + i * 120 * S
            ld.rounded_rectangle([x0, y0, x0 + 210 * S, y0 + 120 * S], radius=34 * S, fill=(198, 124, 56))
            for k in range(4):
                ld.line([(x0 + (35 + 45 * k) * S, y0 + 14 * S), (x0 + (15 + 45 * k) * S, y0 + 106 * S)],
                        fill=(120, 66, 22), width=9 * S)

    def grain(ld):
        for _ in range(170):
            x, y = random.randint(c, c + r), random.randint(c, c + r)
            a = random.random() * 3.14
            ld.ellipse([x - 8 * S, y - 4 * S, x + 8 * S, y + 4 * S], fill=random.choice([(215, 170, 100), (190, 140, 75), (235, 200, 140)]))

    section(90, 270, (200, 230, 201), veg)
    section(270, 360, (255, 224, 178), protein)
    section(0, 90, (255, 236, 200), grain)
    d = ImageDraw.Draw(img)
    d.line([(c, c - r), (c, c + r)], fill=(255, 255, 255), width=7 * S)
    d.line([(c, c), (c + r, c)], fill=(255, 255, 255), width=7 * S)
    d.ellipse([c - r, c - r, c + r, c + r], outline=(255, 255, 255), width=5 * S)

    f = font(30 * S)

    def tag(x, y, text, color):
        tw = d.textlength(text, font=f)
        d.rounded_rectangle([x - tw / 2 - 22 * S, y - 26 * S, x + tw / 2 + 22 * S, y + 26 * S], radius=26 * S, fill=color)
        d.text((x - tw / 2, y - 21 * S), text, font=f, fill=(255, 255, 255))
    tag(c - r * 0.50, c + r * 0.80, "½ Vegetables", (46, 125, 50))
    tag(c + r * 0.52, c - r * 0.82, "¼ Protein", (191, 100, 30))
    tag(c + r * 0.52, c + r * 0.80, "¼ Whole grains", (160, 110, 40))

    # cutlery
    kx = c - R - 70 * S
    d.rounded_rectangle([kx - 8 * S, c - 250 * S, kx + 8 * S, c + 250 * S], radius=8 * S, fill=(176, 190, 197))
    for i in range(3):
        d.rounded_rectangle([kx + 40 * S + i * 0 + (R * 2 + 140 * S) - 40 * S + (i - 1) * 24 * S - 5 * S, c - 250 * S,
                             kx + 40 * S + (R * 2 + 140 * S) - 40 * S + (i - 1) * 24 * S + 5 * S, c - 90 * S], radius=5 * S, fill=(176, 190, 197))
    fx = kx + R * 2 + 140 * S
    d.rounded_rectangle([fx - 8 * S, c - 90 * S, fx + 8 * S, c + 250 * S], radius=8 * S, fill=(176, 190, 197))
    img.resize((W // S, W // S), Image.LANCZOS).save(os.path.join(OUT, "plate.png"), optimize=True)


if __name__ == "__main__":
    hero()
    plate()
    print("assets written to", os.path.abspath(OUT))
