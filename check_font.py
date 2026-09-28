#!/usr/bin/env python3
"""
Check that a font (Nastaliq for Urdu, Naskh for the Arabic matn) can actually draw everything
the dataset needs, and preview how it renders harakat and honorifics. Fonts that lack a glyph draw a blank or a box - training on those
images teaches the model nonsense, so check before generating any synthetic data.

  pip install fonttools pillow
  python check_font.py --font ~/Library/Fonts/JameelNooriNastaleeq.ttf --out font_preview

Prints coverage per character group and of the whole --charset, lists which honorific forms
the font can print (ligature glyph / sign over the name), and writes sample PNGs.
"""
import argparse, unicodedata
from pathlib import Path

NEEDED = {
    "Urdu letters":      "ابپتٹثجچحخدڈذرڑزژسشصضطظعغفقکگلمنںوہھءیےآأؤئ",
    "Harakat":           "\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0670",
    "Arabic-only":       "\u0625\u0649\u0643\u064A\u0647\u0629",
    "Urdu digits":       "۰۱۲۳۴۵۶۷۸۹",
    "Arabic digits":     "٠١٢٣٤٥٦٧٨٩",
    "Punctuation":       "۔،؛؟()[]«»“”‘’",
    "Quranic marks":     "\u06D6\u06DB\u06DD\u06DE",
    "Honorific signs":   "\u0610\u0611\u0612\u0613",     # drawn over a name; labels still use the phrase
    "Honorific ligs":    "\uFDFA\uFDFB" + "".join(chr(c) for c in range(0xFD40, 0xFD50)),
}

PHRASES = {
    "durood":       "رسول اللہ صلی اللہ علیہ وسلم نے فرمایا",
    "radiallahu":   "حضرت ابوبکر رضی اللہ عنہ سے روایت ہے",
    "rahmatullah":  "صدر الشریعہ رحمۃ اللہ علیہ تحریر فرماتے ہیں",
    "alayhissalam": "حضرت موسیٰ علیہ السلام کا واقعہ",
    "bismillah":    "بسم اللہ الرحمن الرحیم",
    "arabic_matn":  "مَنْ قَالَ لِعَالِمٍ عُوَيْلِمٌ اسْتِخْفَافًا فَقَدْ كَفَرَ",
    "digits":       "فتاوی رضویہ ج ۱۰ ص ۳۹۵ ، بہار شریعت ح 9 ص 131",
    "signs":        "نبی اکرم\u0610 حضرت ابوہریرہ\u0613 امام بخاری\u0612 حضرت موسیٰ\u0611",
    "ligatures":    "نبی اکرم \uFDFA حضرت ابوہریرہ \uFD41 جل جلالہ \uFDFB",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", required=True, help="path to a .ttf/.otf Nastaliq font")
    ap.add_argument("--out", default="font_preview", help="folder for sample PNGs")
    ap.add_argument("--size", type=int, default=64, help="font size for previews")
    ap.add_argument("--charset", default="UrduGlyphs_extended.txt", help="glyph list the model uses")
    a = ap.parse_args()

    from fontTools.ttLib import TTFont
    cmap = set()
    tt = TTFont(a.font, fontNumber=0)
    for table in tt["cmap"].tables:
        cmap |= set(table.cmap.keys())

    print(f"font: {Path(a.font).name}  ({len(cmap)} mapped codepoints)\n")
    missing_total = []
    for group, chars in NEEDED.items():
        missing = [c for c in dict.fromkeys(chars) if ord(c) not in cmap]
        status = "all present" if not missing else "MISSING " + " ".join(
            f"U+{ord(c):04X}({c})" for c in missing)
        print(f"  {group:18} {status}")
        missing_total += missing
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from urdu_text import HONORIFICS, load_charset
    charset = [c for c in load_charset(a.charset) if not c.isspace()]
    lacking = [c for c in charset if ord(c) not in cmap]
    print(f"\n  {a.charset}: {len(charset) - len(lacking)}/{len(charset)} glyphs present")
    if lacking:
        print("    lacking: " + " ".join(f"{c}(U+{ord(c):04X})" for c in lacking))
        print("    make_lines.py skips any line this font cannot draw, so a gap only means that")
        print("    those characters must come from another font in the same pool")

    print("\n  honorific forms this font can print (labels always use the full phrase):")
    for phrase, forms in HONORIFICS.items():
        can = [f"ligature {g}" for g in forms["lig"] if ord(g) in cmap]
        if forms["mark"] and ord(forms["mark"]) in cmap:
            can.append(f"sign {forms['mark']}")
        print(f"    {phrase:26} {', '.join(can) or 'small text only'}")
    print("  signs.png / ligatures.png show them; compare with how your books print them")

    # previews
    from PIL import Image, ImageDraw, ImageFont, features
    raqm = features.check("raqm")
    print(f"\nPillow RAQM text shaping: {'available' if raqm else 'NOT available'}")
    if not raqm:
        print("  Without RAQM, Nastaliq will render as disconnected letters and the previews")
        print("  (and any synthetic data) will be useless. Install a Pillow build with libraqm:")
        print("    macOS:  brew install libraqm && pip install --force-reinstall --no-binary :all: pillow")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(a.font, a.size)
    kw = {"language": "ur", "direction": "rtl"} if raqm else {}
    from PIL import ImageOps
    for name, text in PHRASES.items():
        # generous canvas, then crop to the ink: Nastaliq swashes overhang their advance box
        w = int(font.getlength(text, **kw)) + 4 * a.size
        img = Image.new("L", (w, 4 * a.size), 255)
        ImageDraw.Draw(img).text((w - 2 * a.size, int(2.4 * a.size)), text, font=font, fill=0,
                                 anchor="rs", **kw)
        box = ImageOps.invert(img).getbbox()
        if box:
            img = img.crop((box[0] - 20, box[1] - 20, box[2] + 20, box[3] + 20))
        img.save(out / f"{name}.png")
    print(f"\nwrote {len(PHRASES)} preview images to {out}/ - open them and check that:")
    print("  - letters join correctly and the Nastaliq slope looks like your books")
    print("  - the honorific signs / ligatures look like the ones in your books")
    print("  - harakat in arabic_matn.png are visible and correctly placed")


if __name__ == "__main__":
    main()
