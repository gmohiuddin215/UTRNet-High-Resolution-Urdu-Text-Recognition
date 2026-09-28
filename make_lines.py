#!/usr/bin/env python3
"""
Synthetic text-line images + labels for fine-tuning UTRNet on Urdu hadith books.

Built for pages like Sunan Ibn Majah (Urdu): bold fully-vowelled Arabic matn in a Naskh face,
dense Urdu translation in Nastaliq, small honorific clusters, hadith numbers, running headers
with Urdu digits, and footnotes.

What makes it match those pages:
  * separate font pools per script - the matn is NOT Nastaliq on these pages
  * honorifics printed the three ways books print them, always labelled as the full phrase:
      - the ligature glyph (ﷺ, and the Unicode 14 ligatures if a font has them)
      - the sign over the end of the name (ابوہریرہؓ) - what the sample Ibn Majah page uses
      - the phrase in small raised type
  * lines composed from segments, so one line can mix scripts, sizes and weights
  * photocopy-style degradation: speckle, faded/heavy strokes, skew, blur, JPEG
  * every line is checked: glyphs the font lacks (tofu) and labels outside --charset are rejected

  pip install pillow numpy fonttools        # Pillow must have RAQM (see check_font.py)
  python make_lines.py \\
      --urdu_fonts fonts/JameelNooriNastaleeq.ttf \\
      --arabic_fonts fonts/NotoNaskhArabic-Bold.ttf fonts/ScheherazadeNew-Bold.ttf \\
      --urdu_corpus urdu.txt --arabic_corpus matn.txt --n 80000 --imgH 64 --out synth_lines

Output: synth_lines/images/*.png and synth_lines/gt.txt ("images/x.png<TAB>label"),
ready for create_lmdb_dataset.py. Images are unflipped - UTRNet flips them itself.
"""
import argparse, io, random
from pathlib import Path

from urdu_text import HONORIFICS, HARAKAT, clean, load_charset, outside

NAMES = ["رسول اللہ", "نبی اکرم", "حضرت ابوہریرہ", "حضرت ابوبکر", "حضرت عمر", "حضرت عائشہ",
         "امام ابن ماجہ", "امام بخاری", "امام مسلم", "حضرت انس", "حضرت عبد اللہ بن عمر",
         "حضرت موسیٰ", "حضرت علی", "حضرت فاطمہ", "امام ابوحنیفہ"]
AR_HONORIFICS = ["صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ", "رَضِيَ اللَّهُ عَنْهُ", "عَلَيْهِ السَّلَامُ"]
AR_ISNAD = ["حَدَّثَنَا أَبُو بَكْرِ بْنُ أَبِي شَيْبَةَ قَالَ حَدَّثَنَا شَرِيكٌ عَنِ الْأَعْمَشِ عَنْ أَبِي صَالِحٍ",
            "عَنْ أَبِي هُرَيْرَةَ قَالَ قَالَ رَسُولُ اللَّهِ",
            "حَدَّثَنَا مُحَمَّدُ بْنُ الصَّبَّاحِ قَالَ أَنْبَأَنَا جَرِيرٌ عَنِ الْأَعْمَشِ",
            "مَا أَمَرْتُكُمْ بِهِ فَخُذُوهُ وَمَا نَهَيْتُكُمْ عَنْهُ فَانْتَهُوا",
            "مَنْ أَطَاعَنِي فَقَدْ أَطَاعَ اللَّهَ وَمَنْ عَصَانِي فَقَدْ عَصَى اللَّهَ",
            "إِنَّمَا هَلَكَ مَنْ كَانَ قَبْلَكُمْ بِسُؤَالِهِمْ وَاخْتِلَافِهِمْ عَلَى أَنْبِيَائِهِمْ"]
HEADERS = ["سنن ابن ماجہ (جزء اول)", "کتاب السنۃ", "باب: اتباع سنۃ رسول اللہ", "کتاب الطہارۃ",
           "سنن ابو داؤد (جلد دوم)", "مشکوٰۃ شریف", "بخاری شریف (جلد اول)"]
UR_TAIL = ["نے ارشاد فرمایا:", "روایت کرتے ہیں", "سے روایت ہے کہ", "فرماتے ہیں کہ",
           "کی سنت کی پیروی کرنا", "تحریر فرماتے ہیں"]
VOWELS = "ًٌٍَُِّْ"
DIGITS = ["0123456789",
          "٠١٢٣٤٥٦٧٨٩",
          "۰۱۲۳۴۵۶۷۸۹"]


def num(rng, lo=1, hi=999, system=None):
    d = system if system is not None else rng.choice(DIGITS)
    return "".join(d[int(c)] for c in str(rng.randint(lo, hi)))


def vowelize(text, rng, rate=0.85):
    """Add random harakat to unvowelled text (never to letters that already carry a mark)."""
    letters = sum("ء" <= c <= "ي" for c in text)
    if not letters or sum(c in HARAKAT for c in text) / letters > 0.2:
        return text                                              # already vowelled: keep as is
    out = []
    for i, ch in enumerate(text):
        out.append(ch)
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if ("ء" <= ch <= "ي" and ch not in "اوي" and nxt not in HARAKAT
                and rng.random() < rate):
            out.append(rng.choice(VOWELS))
    return "".join(out)


def read_corpus(path, min_len=8):
    if not path:
        return []
    return [s for s in (clean(l) for l in Path(path).read_text(encoding="utf-8").split("\n"))
            if len(s) >= min_len]


def span(words, rng, lo=3, hi=12):
    if not words:
        return ""
    n = rng.randint(lo, hi)
    i = rng.randrange(max(1, len(words) - n))
    return " ".join(words[i:i + n])


# ---------------------------------------------------------------- line composition
def seg(text, script="ur", scale=1.0, bold=False, hon=None):
    return {"text": text, "script": script, "scale": scale, "bold": bold, "hon": hon}


# segments are listed in reading order (right to left on the page)
def compose(kind, urdu_words, arabic_lines, rng):
    if kind == "urdu_honorific":
        segs = [seg(rng.choice(NAMES)), seg("", hon=rng.choice(list(HONORIFICS))),
                seg(" " + (span(urdu_words, rng, 3, 8) or rng.choice(UR_TAIL)))]
        if rng.random() < 0.35:
            segs.insert(0, seg(num(rng, 1, 400, DIGITS[0]) + "- ", bold=True))
        if rng.random() < 0.4:                                   # a second honorific later in the line
            segs += [seg(" " + rng.choice(NAMES)), seg("", hon=rng.choice(list(HONORIFICS))),
                     seg(" " + rng.choice(UR_TAIL))]
        return segs
    if kind == "arabic_matn":
        base = rng.choice(arabic_lines) if arabic_lines and rng.random() < 0.7 else rng.choice(AR_ISNAD)
        txt = vowelize(base, rng, rng.uniform(0.6, 1.0)) if rng.random() < 0.7 else base
        segs = [seg(txt, "ar", 1.0, rng.random() < 0.6)]
        if rng.random() < 0.4:
            segs.insert(0, seg(num(rng, 1, 400, DIGITS[0]) + "- ", "ar", 1.0, True))
        if rng.random() < 0.3:
            segs.append(seg(" " + rng.choice(AR_HONORIFICS), "ar", rng.uniform(0.6, 1.0)))
        return segs
    if kind == "mixed":
        ar = vowelize(rng.choice(arabic_lines) if arabic_lines else rng.choice(AR_ISNAD), rng, 0.8)
        ur = span(urdu_words, rng, 3, 7) or "ترجمہ یہ ہے"
        return ([seg(ar, "ar", 1.0, True), seg(" " + ur)] if rng.random() < 0.5 else
                [seg(ur + " "), seg(ar, "ar", 1.0, True)])
    if kind == "header":
        return [seg(rng.choice(HEADERS) + "   " + num(rng, 1, 600, DIGITS[2]))]
    if kind == "footnote":
        return [seg(f"{num(rng, 1, 9, DIGITS[0])} : " + (span(urdu_words, rng, 4, 10) or
                    "اس روایت کو نقل کرنے میں امام ابن ماجہ منفرد ہیں۔"), scale=0.8)]
    if kind == "reference":
        return [seg(f"({rng.choice(HEADERS)} ج {num(rng, 1, 30)} ص {num(rng, 1, 600)})")]
    return [seg(span(urdu_words, rng, 4, 12) or rng.choice(NAMES))]


def label_of(segments):
    return clean("".join(f" {s['hon']} " if s["hon"] else s["text"] for s in segments))


# ---------------------------------------------------------------- fonts
class FontPool:
    """Cached ImageFont objects and cmaps."""
    def __init__(self):
        self.fonts, self.cmaps = {}, {}

    def get(self, path, size):
        key = (path, size)
        if key not in self.fonts:
            from PIL import ImageFont
            self.fonts[key] = ImageFont.truetype(path, size)
        return self.fonts[key]

    def covers(self, path, text):
        if path not in self.cmaps:
            from fontTools.ttLib import TTFont
            cmap = set()
            for table in TTFont(path, fontNumber=0)["cmap"].tables:
                cmap |= set(table.cmap.keys())
            self.cmaps[path] = cmap
        return all(ord(c) in self.cmaps[path] for c in text if not c.isspace())


def plan(segments, pool, fonts, base, rng):
    """Resolve fonts, sizes and the printed form of each honorific.
    Returns [(draw_text, font_path, size, bold, raise_px, language)] or None if a glyph is missing."""
    pieces = []
    for s in segments:
        lang = "ur" if s["script"] == "ur" else "ar"
        pool_paths = fonts[s["script"]]
        if s["hon"]:
            forms = HONORIFICS[s["hon"]]
            options = []                                         # (weight, kind, text)
            for lig in forms["lig"]:
                options.append((0.45, "lig", lig))
            prev = pieces[-1] if pieces else None
            if forms["mark"] and prev and prev[5] == "ur" and prev[0].rstrip() == prev[0]:
                options.append((0.35, "mark", forms["mark"]))
            options.append((0.20, "text", s["hon"]))
            rng.shuffle(options)
            for _, kind, text in sorted(options, key=lambda o: -o[0] * rng.random()):
                if kind == "mark":
                    if pool.covers(prev[1], text):
                        pieces[-1] = (prev[0] + text,) + prev[1:]
                        break
                    continue
                path = next((p for p in [pool_paths[0]] + fonts["ur"] + fonts["ar"]
                             if pool.covers(p, text)), None)
                if path is None:
                    continue
                if kind == "lig":
                    size = int(base * rng.uniform(0.7, 1.0))
                    lift = int(base * rng.uniform(0.0, 0.3))
                    pieces.append((" " + text + " ", path, size, False, lift, lang))
                else:
                    size = max(10, int(base * rng.uniform(0.40, 0.65)))
                    lift = int(base * rng.uniform(0.2, 0.5))
                    pieces.append((" " + text + " ", path, size, False, lift, lang))
                break
            else:
                return None
            continue
        if not s["text"]:
            continue
        path = next((p for p in pool_paths if pool.covers(p, s["text"])), None)
        if path is None:
            return None                                          # would print tofu: skip line
        pieces.append((s["text"], path, max(10, int(base * s["scale"])), s["bold"], 0, lang))
    return pieces or None


def render(pieces, pool, base, rng):
    """Draw pieces right to left on a generous canvas, then crop to the ink."""
    from PIL import Image, ImageDraw, ImageOps
    widths = []
    for text, path, size, bold, _, lang in pieces:
        font = pool.get(path, size)
        widths.append(font.getlength(text, direction="rtl", language=lang) + 2 * int(bold))
    W, H = int(sum(widths) + 4 * base), int(4 * base)
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)
    x, baseline, ink = W - 2 * base, int(2.4 * base), rng.randint(0, 50)
    for (text, path, size, bold, lift, lang), w in zip(pieces, widths):
        d.text((x, baseline - lift), text, font=pool.get(path, size), fill=ink, anchor="rs",
               direction="rtl", language=lang, stroke_width=int(bold), stroke_fill=ink)
        x -= w
    box = ImageOps.invert(img).getbbox()
    if not box:
        return None
    px, py = rng.randint(6, 20), rng.randint(4, 14)
    return img.crop((max(0, box[0] - px), max(0, box[1] - py), min(W, box[2] + px), min(H, box[3] + py)))


def degrade(img, rng, nrng, allow_thin=True, gentle=False):
    """gentle: the line carries harakat - keep blur and contrast loss mild so the marks survive"""
    from PIL import Image, ImageFilter, ImageEnhance
    import numpy as np
    if rng.random() < 0.7:
        img = img.rotate(rng.uniform(-1.4, 1.4), expand=True, fillcolor=255, resample=Image.BICUBIC)
    # heavy (MinFilter) or thin (MaxFilter) strokes; thinning only on its own, because thin + pale +
    # low contrast together erases the text while the label still says it is there
    thinned = False
    if rng.random() < 0.5:
        thinned = allow_thin and rng.random() < 0.35
        img = img.filter(ImageFilter.MaxFilter(3) if thinned else ImageFilter.MinFilter(3))
    if rng.random() < 0.8:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8 if gentle or thinned else 1.3)))
    if rng.random() < 0.6:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.9 if gentle or thinned else 0.7, 1.3))
    a = np.asarray(img).astype("float32")
    if rng.random() < 0.8:
        a += nrng.normal(0, rng.uniform(4, 16), a.shape)
    if rng.random() < 0.5:
        m = nrng.random(a.shape) < rng.uniform(0.0005, 0.004)
        a[m] = nrng.choice([0.0, 255.0], int(m.sum()))
    img = Image.fromarray(a.clip(0, 255).astype("uint8"))
    if rng.random() < 0.5:
        buf = io.BytesIO(); img.save(buf, "JPEG", quality=rng.randint(30, 85)); buf.seek(0)
        img = Image.open(buf).convert("L")
    return img


def shorten(segments, attempt):
    """Drop words from the longest plain segment, keeping its surrounding spaces."""
    plain = [s for s in segments if not s["hon"] and s["text"].strip()]
    if not plain:
        return segments
    longest = max(plain, key=lambda s: len(s["text"]))
    t = longest["text"]
    lead, trail = t[:len(t) - len(t.lstrip())], t[len(t.rstrip()):]
    words = t.split()
    keep = max(2, len(words) * (4 - attempt) // 4)
    return [dict(s, text=lead + " ".join(words[:keep]) + trail) if s is longest else s for s in segments]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdu_fonts", nargs="+", required=True, help="Nastaliq .ttf/.otf")
    ap.add_argument("--arabic_fonts", nargs="+", required=True, help="Naskh .ttf/.otf for matn")
    ap.add_argument("--urdu_corpus")
    ap.add_argument("--arabic_corpus")
    ap.add_argument("--charset", default="UrduGlyphs_extended.txt",
                    help="labels with characters outside this list are rejected")
    ap.add_argument("--out", default="synth_lines")
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--imgH", type=int, default=64)
    ap.add_argument("--max_width", type=int, default=0,
                    help="longest allowed line after resizing (0 = 36 x imgH, like a full printed line)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--p_urdu_honorific", type=float, default=0.25)
    ap.add_argument("--p_arabic_matn", type=float, default=0.35)
    ap.add_argument("--p_mixed", type=float, default=0.10)
    ap.add_argument("--p_header", type=float, default=0.04)
    ap.add_argument("--p_footnote", type=float, default=0.06)
    ap.add_argument("--p_reference", type=float, default=0.05)
    a = ap.parse_args()

    import numpy as np
    from PIL import Image, features
    if not features.check("raqm"):
        raise SystemExit("Pillow lacks RAQM: Arabic/Nastaliq would render unjoined. See check_font.py.")

    rng, nrng = random.Random(a.seed), np.random.default_rng(a.seed)
    charset = load_charset(a.charset)
    max_width = a.max_width or 36 * a.imgH
    urdu_words = " ".join(read_corpus(a.urdu_corpus)).split()
    arabic_lines = read_corpus(a.arabic_corpus)
    print(f"corpus: {len(urdu_words):,} Urdu words, {len(arabic_lines):,} Arabic lines")

    kinds = (["urdu_honorific"] * int(a.p_urdu_honorific * 100) +
             ["arabic_matn"] * int(a.p_arabic_matn * 100) +
             ["mixed"] * int(a.p_mixed * 100) +
             ["header"] * int(a.p_header * 100) +
             ["footnote"] * int(a.p_footnote * 100) +
             ["reference"] * int(a.p_reference * 100))
    kinds += ["urdu"] * max(0, 100 - len(kinds))

    pool = FontPool()
    out = Path(a.out); (out / "images").mkdir(parents=True, exist_ok=True)
    gt, by_kind = [], {}
    rejected = {"width": 0, "font lacks glyph": 0, "outside charset": 0}
    bad_chars = set()
    for i in range(a.n):
        kind = rng.choice(kinds)
        fonts = {"ur": rng.sample(a.urdu_fonts, len(a.urdu_fonts)),
                 "ar": rng.sample(a.arabic_fonts, len(a.arabic_fonts))}
        segs = compose(kind, urdu_words, arabic_lines, rng)
        label = label_of(segs)
        missing = outside(label, charset)
        if missing:
            rejected["outside charset"] += 1
            bad_chars.update(missing)
            continue
        vowelled = sum(c in HARAKAT for c in label) > 0.1 * max(1, len(label))
        reason = "width"
        for attempt in range(4):
            if attempt:
                segs = shorten(segs, attempt)
                label = label_of(segs)
            base = rng.choice([56, 64, 72] if vowelled else [40, 48, 56, 64, 72])
            pieces = plan(segs, pool, fonts, base, rng)
            if pieces is None:
                reason = "font lacks glyph"
                break
            img = render(pieces, pool, base, rng)
            if img is None:
                break
            clean_img = img
            img = degrade(clean_img, rng, nrng, allow_thin=base >= 56 and not vowelled, gentle=vowelled)
            if np.percentile(np.asarray(img), 2) > 140:              # text washed out: degrade gently
                img = degrade(clean_img, rng, nrng, allow_thin=False)
                if np.percentile(np.asarray(img), 2) > 140:
                    img = clean_img
            img = img.resize((max(8, round(img.width * a.imgH / img.height)), a.imgH), Image.LANCZOS)
            if img.width <= max_width:
                name = f"images/{i:07d}.png"
                img.save(out / name)
                gt.append(f"{name}\t{label}")
                by_kind[kind] = by_kind.get(kind, 0) + 1
                reason = None
                if len(gt) % 5000 == 0:
                    print(f"  {len(gt):,}/{a.n:,}", flush=True)
                break
        if reason:
            rejected[reason] += 1

    (out / "gt.txt").write_text("\n".join(gt) + "\n", encoding="utf-8")
    print(f"\nwrote {len(gt):,} lines to {out}/")
    for k, v in sorted(by_kind.items(), key=lambda x: -x[1]):
        print(f"  {k:16} {v:,}")
    for k, v in rejected.items():
        if v:
            print(f"  rejected ({k}): {v:,}")
    if bad_chars:
        print("  characters outside --charset: " + " ".join(f"{c}(U+{ord(c):04X})" for c in sorted(bad_chars)))
    if rejected["font lacks glyph"]:
        print("  run check_font.py on your fonts to see which glyphs they lack")
    print(f"next: python create_lmdb_dataset.py --inputPath {out} --gtFile {out}/gt.txt --outputPath lmdb/synth")


if __name__ == "__main__":
    main()
