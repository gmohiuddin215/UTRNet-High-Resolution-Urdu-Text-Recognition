#!/usr/bin/env python3
"""
OCR book pages on your Mac: page images or PDFs in, Urdu/Arabic text out.

  python ocr_page.py book.pdf --model UrduOCR.mlpackage                # Core ML, same as the app
  python ocr_page.py scans/ --model results/utrnet_hadith_final.pth \\
      --charset results/UrduGlyphs_extended.txt                         # PyTorch
  python ocr_page.py book.pdf --pages 10-20 --preview                  # also save line boxes

Each page is deskewed (up to 15 degrees either way), split into lines, and every line is read
by the model. Text goes to ocr_output/<name>.txt (one printed line per text line, pages
separated by a "--- page N ---" line) and to the terminal.

--model takes the .mlpackage from export_coreml.py (reads exactly as the iPhone/Mac app does;
the charset is inside) or the .pth checkpoint (needs --charset). Both give the same text: the
.pth is run through deploy_model.DeployModel, the same deterministic network that was exported.

--preview saves ocr_output/<name>_pN.png with the deskewed page and numbered line boxes: look at
one when a page reads badly. Lines must be found one by one; two-column pages, photos with a
dark background or strong page curl will need cropping first (on the phone, the document
camera does this).

--dump prints the deskew angle, line boxes and a checksum of every model input instead of plain
text; the Swift CLI (ios/UrduOCR, `urdu-ocr --dump`) prints the same format, so `diff` shows
whether the app prepares a page exactly like this script.
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


def page_range(spec, n):
    if not spec:
        return list(range(n))
    pages = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        lo = int(lo)
        hi = int(hi) if hi else lo
        pages.update(range(lo - 1, min(hi, n)))
    return sorted(p for p in pages if 0 <= p < n)


def iter_pages(inputs, dpi, pages):
    """(label, output stem, uint8 gray page) for every page of every input."""
    from ocr_pipeline import load_gray
    files = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT | {".pdf"})
        else:
            files.append(p)
    for f in files:
        if f.suffix.lower() == ".pdf":
            try:
                import pymupdf
            except ImportError:
                import fitz as pymupdf
            doc = pymupdf.open(f)
            for i in page_range(pages, doc.page_count):
                pix = doc[i].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
                gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)
                yield f"{f.name}#{i + 1}", f.stem, gray[:, :pix.width].copy()
        else:
            yield f.name, f.stem, load_gray(f)


class Reader:
    """Runs line tensors through the .mlpackage (Core ML) or the .pth (PyTorch)."""

    def __init__(self, model, charset_path, units):
        self.model_path = model
        if model.rstrip("/").endswith((".mlpackage", ".mlmodel")):
            import json
            import coremltools as ct
            unit = {"cpu": ct.ComputeUnit.CPU_ONLY, "gpu": ct.ComputeUnit.CPU_AND_GPU,
                    "all": ct.ComputeUnit.ALL}[units]
            self.ml = ct.models.MLModel(model, compute_units=unit)
            meta = self.ml.user_defined_metadata
            self.charset = json.loads(meta["charset"]) if "charset" in meta else None
            if charset_path:
                self.charset = list(_load_charset(charset_path))
            if self.charset is None:
                raise SystemExit("this .mlpackage has no charset inside: pass --charset")
            # lines per call, fixed when the package was exported (export_coreml.py --batch)
            self.batch = int(self.ml.get_spec().description.input[0].type.multiArrayType.shape[0])
            self.ml.predict({"image": np.zeros((self.batch, 1, 64, 800), np.float32)})  # load now, not in page 1's time
            self.torch = None
        else:
            if not charset_path:
                raise SystemExit("--charset is required with a .pth checkpoint")
            import torch
            from deploy_model import load_deploy
            torch.set_grad_enabled(False)
            self.charset = list(_load_charset(charset_path))
            self.torch = load_deploy(model, "".join(self.charset))
            self.batch = 8

    def read(self, xs):
        """xs: list of [1, 64, 800] float32 line tensors -> list of texts"""
        from ocr_pipeline import ctc_decode
        texts = []
        for s in range(0, len(xs), self.batch):
            chunk = list(xs[s:s + self.batch])
            n = len(chunk)
            if self.torch is not None:
                import torch
                logits = self.torch(torch.from_numpy(np.stack(chunk))).numpy()
            else:
                chunk += [chunk[-1]] * (self.batch - n)          # the package wants a full batch
                logits = self.ml.predict({"image": np.stack(chunk)})["logits"]
            texts += [ctc_decode(lg.argmax(1), self.charset) for lg in logits[:n]]
        return texts


def _load_charset(path):
    from urdu_text import load_charset
    return load_charset(path)


def save_preview(page, k, lines, path):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(page).convert("RGB")
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default(size=max(16, page.shape[0] // 60))
    except TypeError:                                    # Pillow < 10.1
        font = ImageFont.load_default()
    for n, ((t, b, l, r), _, _) in enumerate(lines):
        col = (220, 40, 40) if n % 2 == 0 else (40, 90, 220)
        d.rectangle([l, t, r - 1, b - 1], outline=col, width=3)
        d.text((max(0, l - 10 * len(str(n + 1)) - 12), t + 2), str(n + 1), fill=col, font=font)
    d.text((10, 10), f"deskew {k * 0.05:+.2f} deg", fill=(0, 140, 0), font=font)
    im.thumbnail((1400, 2000))
    im.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="page images, PDFs, or folders of them")
    ap.add_argument("--model", required=True, help="UrduOCR.mlpackage or the .pth checkpoint")
    ap.add_argument("--charset", help="charset file (required for .pth; the .mlpackage has its own)")
    ap.add_argument("--out", default="ocr_output", help="folder for the .txt files")
    ap.add_argument("--dpi", type=int, default=300, help="PDF render resolution")
    ap.add_argument("--pages", help="PDF pages to read, e.g. 5-12,20 (1-based; default all)")
    ap.add_argument("--no_deskew", action="store_true")
    ap.add_argument("--preview", action="store_true", help="save each page with its line boxes")
    ap.add_argument("--units", choices=["cpu", "gpu", "all"], default="cpu",
                    help="Core ML compute units (cpu = exact float32 reference; default)")
    ap.add_argument("--dump", action="store_true", help="print angle/boxes/input checksums (for diff with Swift)")
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ocr_pipeline import prepare_page

    reader = Reader(a.model, a.charset, a.units)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    texts = {}
    for label, stem, gray in iter_pages(a.inputs, a.dpi, a.pages):
        t0 = time.time()
        k, page, lines = prepare_page(gray, deskew=not a.no_deskew)
        t1 = time.time()
        read = reader.read([x for _, x, _ in lines])
        t2 = time.time()
        if a.dump:
            print(f"page {label} skew {k} size {page.shape[1]}x{page.shape[0]} lines {len(lines)}")
            for i, ((t, b, l, r), _, small) in enumerate(lines):
                sha = hashlib.sha1(np.ascontiguousarray(small).tobytes()).hexdigest()
                print(f"line {i + 1} box {t} {b} {l} {r} width {small.shape[1]} sha1 {sha}")
            for i, text in enumerate(read):
                print(f"text {i + 1} {text}")
            continue
        print(f"\n=== {label}  (deskew {k * 0.05:+.2f} deg, {len(lines)} lines; "
              f"page prep {t1 - t0:.1f}s, reading {t2 - t1:.1f}s)")
        print("\n".join(read))
        part = label.rpartition("#")[2] if "#" in label else None
        texts.setdefault(stem, []).append((part, read))
        if a.preview:
            save_preview(page, k, lines, out / (f"{stem}_p{part}.png" if part else f"{stem}.png"))
    for stem, pages in texts.items():
        chunks = [(f"--- page {part} ---\n" if part else "") + "\n".join(lines) for part, lines in pages]
        (out / f"{stem}.txt").write_text("\n\n".join(chunks) + "\n", encoding="utf-8")
    if texts:
        print(f"\ntext written to {out}/: " + ", ".join(f"{s}.txt" for s in texts))


if __name__ == "__main__":
    main()
