#!/usr/bin/env python3
"""
Turn corrected book pages into labeled line images for UTRNet.

Input : page.png  +  page.txt   (your corrected text, one printed line per text line)
Output: line crops + gt.txt, ready for create_lmdb_dataset.py

  pip install pillow numpy
  # 1. look at the segmentation first, without writing a dataset
  python page_to_lines.py --data my_pages --preview_only
  # 2. then cut and label
  python page_to_lines.py --data my_pages --out real_lines --charset UrduGlyphs_extended.txt

For each page it deskews, erases horizontal rules, finds the text lines and pairs line N with
text line N of page.txt. Pages where the counts differ are skipped and listed, so you can fix
them instead of silently training on misaligned data.

Writing page.txt
  * one printed line = one text line, in reading order, header and footnotes included
  * a line containing only "#" keeps the position but skips that band (e.g. a header you
    do not want to transcribe)
  * honorifics: type the full phrase (صلی اللہ علیہ وسلم) or the sign (ﷺ ؓ ؒ ...), both are
    stored as the full phrase; text is NFC-normalised and zero-width characters are removed

Preview images (page_previews/) number every detected line - check a few before trusting the
output. If two printed lines share one box, lower --split; if one line is cut in two, raise
--min_gap.
"""
import argparse
from pathlib import Path

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def otsu(gray):
    import numpy as np
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (1 - omega)
    denom[denom == 0] = 1e-9
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    # Otsu's class 0 is "value <= t"; return t + 1 so callers can test `gray < otsu(gray)`.
    # On a black-and-white scan every t from 0 to 254 scores the same: take the middle of the
    # tied range, or nothing would count as ink (t = 0 means only pixels below 0)
    best = np.flatnonzero(sigma_b >= np.nanmax(sigma_b) - 1e-9)
    return int(best[0] + best[-1]) // 2 + 1


def deskew(gray, limit=2.0, step=0.25):
    """Rotate by the small angle that makes the horizontal projection most peaked."""
    import numpy as np
    from PIL import Image
    best, best_score = 0.0, -1
    ink = (gray < otsu(gray)).astype("float32")
    src = Image.fromarray((ink * 255).astype("uint8"))
    a = -limit
    while a <= limit + 1e-9:
        rot = np.asarray(src.rotate(a, resample=Image.BILINEAR, fillcolor=0)) > 127
        proj = rot.sum(axis=1).astype("float32")
        score = float(((proj[1:] - proj[:-1]) ** 2).sum())      # sharper bands = higher score
        if score > best_score:
            best, best_score = a, score
        a += step
    return best


def erase_rules(ink, min_frac, margin=3):
    """Clear horizontal runs of ink longer than min_frac of the page width (rules, underlines),
    plus a few rows around them so the ragged edges of a thick or double rule go too."""
    import numpy as np
    out = ink.copy()
    min_len = int(min_frac * ink.shape[1])
    for y in np.where(ink.sum(axis=1) >= min_len)[0]:
        row = np.concatenate(([0], ink[y].astype("int8"), [0]))
        edges = np.flatnonzero(np.diff(row))
        for s, e in zip(edges[::2], edges[1::2]):
            if e - s >= min_len:
                out[max(0, y - margin):y + margin + 1, s:e] = False
    return out


def runs(mask):
    """(start, end) of consecutive True runs."""
    out, start = [], None
    for y, v in enumerate(mask):
        if v and start is None:
            start = y
        elif not v and start is not None:
            out.append((start, y))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def find_lines(ink, min_gap, min_height, split, low, strong=0.5):
    """Text lines as (top, bottom) crop limits.

    Two thresholds: a low one finds every region with text (so short last lines and small
    footnotes are kept, harakat included), a high one finds the dense core of each line.
    A region with several cores holds several lines and is cut at the emptiest row between them.
    """
    import numpy as np
    proj = ink.sum(axis=1).astype("float32")
    if proj.max() == 0:
        return []
    k = max(3, ink.shape[0] // 400) | 1
    smooth = np.convolve(proj, np.ones(k) / k, mode="same")
    ref = float(np.percentile(smooth[smooth > 0], 90))

    regions = []
    for r in runs(smooth > max(1.0, low * ref)):                 # merge tiny gaps (split dots)
        if regions and r[0] - regions[-1][1] < min_gap:
            regions[-1] = (regions[-1][0], r[1])
        else:
            regions.append(r)

    lines = []
    for top, bottom in regions:
        if bottom - top < min_height:
            continue                                             # specks, page-edge dust
        seg = smooth[top:bottom]
        # a core only starts a new line if it is as dense as body text; rows of harakat or the
        # dots above a large title are weaker and stay with the line they belong to
        cores = [c for c in runs(seg > split * ref)
                 if c[1] - c[0] >= max(3, min_height // 4) and seg[c[0]:c[1]].max() >= strong * ref]
        if len(cores) <= 1:
            lines.append((top, bottom))
            continue
        cuts = [top]
        for (_, e1), (s2, _) in zip(cores, cores[1:]):
            cuts.append(top + e1 + int(np.argmin(seg[e1:s2])))
        cuts.append(bottom)
        lines += [(a, b) for a, b in zip(cuts, cuts[1:]) if b - a >= min_height]
    return lines


def read_labels(txt):
    from urdu_text import clean
    rows = [l.strip() for l in txt.read_text(encoding="utf-8").split("\n")]
    return [r if r == "#" else clean(r) for r in rows if r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder of page images + matching .txt files")
    ap.add_argument("--out", default="real_lines")
    ap.add_argument("--imgH", type=int, default=0,
                    help="resize crops to this height (0 = keep scan resolution; train.py resizes anyway)")
    ap.add_argument("--charset", default="UrduGlyphs_extended.txt",
                    help="lines with characters outside this list are reported and skipped")
    ap.add_argument("--pad", type=int, default=4, help="extra pixels above/below each line")
    ap.add_argument("--min_gap", type=int, default=0, help="merge text regions closer than this (px; 0 = auto)")
    ap.add_argument("--min_height", type=int, default=0, help="ignore regions shorter than this (px; 0 = auto)")
    ap.add_argument("--split", type=float, default=0.12,
                    help="core threshold as a share of a typical line's ink; lower splits more")
    ap.add_argument("--low", type=float, default=0.02, help="region threshold as a share of a typical line's ink")
    ap.add_argument("--strong", type=float, default=0.5,
                    help="a dense core must reach this share of a typical line's ink to count as its own line")
    ap.add_argument("--rule", type=float, default=0.15,
                    help="erase horizontal strokes longer than this share of the page width")
    ap.add_argument("--no_deskew", action="store_true")
    ap.add_argument("--preview_only", action="store_true", help="only write previews, no dataset")
    a = ap.parse_args()

    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from urdu_text import load_charset, outside

    charset = load_charset(a.charset) if not a.preview_only else ""
    root = Path(a.data)
    pages = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT and p.is_file())
    if not pages:
        raise SystemExit(f"No page images under {root}")
    out = Path(a.out)
    prev_dir = out / "page_previews"
    (out / "images").mkdir(parents=True, exist_ok=True)
    prev_dir.mkdir(parents=True, exist_ok=True)

    gt, mismatched, bad_chars, n_pages = [], [], [], 0
    for page in pages:
        img = Image.open(page).convert("L")
        angle = 0.0 if a.no_deskew else deskew(np.asarray(img.resize(
            (img.width // 4, img.height // 4))))
        if angle:
            img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=255, expand=True)
        gray = np.asarray(img)
        ink = erase_rules(gray < otsu(gray), a.rule)

        min_gap = a.min_gap or max(4, img.height // 300)
        min_height = a.min_height or max(12, img.height // 120)
        lines = find_lines(ink, min_gap, min_height, a.split, a.low, a.strong)
        lines = [(max(0, t - a.pad), min(img.height, b + a.pad)) for t, b in lines]

        prev = img.convert("RGB")
        d = ImageDraw.Draw(prev)
        try:
            label_font = ImageFont.load_default(size=max(16, img.height // 60))
        except TypeError:                                        # Pillow < 10.1
            label_font = ImageFont.load_default()
        for n, (t, b) in enumerate(lines):                       # alternate colours per line
            col = (220, 40, 40) if n % 2 == 0 else (40, 90, 220)
            d.rectangle([2, t, prev.width - 3, b - 1], outline=col, width=3)
            d.text((8, t + 2), str(n + 1), fill=col, font=label_font)
        prev.thumbnail((1000, 1400))
        prev.save(prev_dir / f"{page.stem}.png")

        if a.preview_only:
            print(f"{page.name}: {len(lines)} lines (deskew {angle:+.2f}deg)")
            continue
        txt = page.with_suffix(".txt")
        if not txt.exists():
            print(f"{page.name}: no .txt - skipped")
            continue
        labels = read_labels(txt)
        if len(labels) != len(lines):
            mismatched.append((page.name, len(lines), len(labels)))
            continue
        n_pages += 1
        for i, ((t, b), text) in enumerate(zip(lines, labels)):
            if text == "#":
                continue
            missing = outside(text, charset)
            if missing:
                bad_chars.append((page.name, i + 1, "".join(missing)))
                continue
            crop = img.crop((0, t, img.width, b))
            cols = np.where(ink[t:b].any(axis=0))[0]             # trim side margins
            if len(cols):
                crop = crop.crop((max(0, cols[0] - 8), 0, min(crop.width, cols[-1] + 8), crop.height))
            if a.imgH:
                crop = crop.resize((max(8, round(crop.width * a.imgH / crop.height)), a.imgH), Image.LANCZOS)
            name = f"images/{page.stem}_{i + 1:03d}.png"
            crop.save(out / name)
            gt.append(f"{name}\t{text}")

    if a.preview_only:
        print(f"\npreviews in {prev_dir}/ - open a few and check every printed line got its own box")
        return

    (out / "gt.txt").write_text("\n".join(gt) + "\n", encoding="utf-8")
    print(f"\n{len(gt):,} labeled lines from {n_pages} pages -> {out}/")
    if mismatched:
        print(f"\n{len(mismatched)} page(s) skipped because detected lines != text lines:")
        for name, nb, nl in mismatched[:20]:
            print(f"  {name}: {nb} lines on the page vs {nl} in the .txt")
        print("  open their previews: usually two lines merged, a header/footnote was missed,")
        print("  or the .txt wraps one printed line onto two. Fix and rerun.")
    if bad_chars:
        print(f"\n{len(bad_chars)} line(s) skipped: characters outside {a.charset}")
        for name, n, chars in bad_chars[:20]:
            print(f"  {name} line {n}: {' '.join(chars)}  " + " ".join(f"U+{ord(c):04X}" for c in chars))
    print(f"\nnext: python create_lmdb_dataset.py --inputPath {out} --gtFile {out}/gt.txt "
          f"--outputPath lmdb/real")


if __name__ == "__main__":
    main()
