#!/usr/bin/env python3
"""
Measure a UTRNet checkpoint on labeled line images, split by what actually goes wrong on
Urdu hadith books: Urdu vs Arabic lines, harakat vs letters, honorifics.

  python eval_lines.py --gt real_lines/gt.txt --root real_lines \\
      --saved_model saved_models/stage2/best_norm_ED.pth --charset UrduGlyphs_extended.txt \\
      --imgH 64 --imgW 800

  # the "free test": same pretrained model, only the input height changes
  python eval_lines.py --gt real_lines/gt.txt --root real_lines --saved_model UTRNet-Large.pth \\
      --charset UrduGlyphs.txt --imgH 32 --imgW 400
  python eval_lines.py ... --imgH 64 --imgW 800

--gt is the usual "image<TAB>label" file (page_to_lines.py / make_lines.py output).
Prints CER per group, CER with harakat ignored (so you can see whether errors are marks or
letters), exact honorific phrases, and the worst lines. --out writes every line to a TSV.

Labels and predictions are both normalised with urdu_text.clean() before scoring.
"""
import argparse
import random
from pathlib import Path

from urdu_text import clean, edit_distance, honorifics_in, is_arabic, strip_harakat, HARAKAT


# Letters that print identically in these fonts (medial ي/ی, ك/ک, ة/ۃ) and word spacing, which
# Urdu print does not show reliably. The lenient CER ignores both, so it counts only what a
# reader could actually see as wrong.
_LOOKALIKE = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ة": "ۃ"})


def lenient(text):
    return text.translate(_LOOKALIKE).replace(" ", "")


def cer(pairs):
    """Character error rate over (reference, prediction) pairs: total edits / total ref chars."""
    chars = sum(len(r) for r, _ in pairs)
    return sum(edit_distance(r, p) for r, p in pairs) / chars if chars else float("nan")


def score(rows):
    """rows: [(image, reference, prediction)] -> printable report lines."""
    groups = {
        "all lines": rows,
        "Urdu lines": [r for r in rows if not is_arabic(r[1])],
        "Arabic lines": [r for r in rows if is_arabic(r[1])],
        "lines with harakat": [r for r in rows if any(c in HARAKAT for c in r[1])],
        "lines with honorifics": [r for r in rows if honorifics_in(r[1])],
    }
    out = [f"{'group':24} {'lines':>6} {'CER':>8} {'CER no harakat':>15} {'lenient CER':>12} {'line acc':>9}"]
    for name, g in groups.items():
        if not g:
            continue
        pairs = [(r, p) for _, r, p in g]
        bare = [(strip_harakat(r), strip_harakat(p)) for r, p in pairs]
        loose = [(lenient(r), lenient(p)) for r, p in pairs]
        acc = sum(r == p for r, p in pairs) / len(pairs)
        out.append(f"{name:24} {len(g):6d} {cer(pairs):8.2%} {cer(bare):15.2%} {cer(loose):12.2%} {acc:9.1%}")

    total = hit = 0
    for _, ref, pred in rows:
        wanted = honorifics_in(ref)
        got = honorifics_in(pred)
        total += len(wanted)
        for h in wanted:                                         # count each phrase at most as often
            if h in got:                                         # as the prediction contains it
                got.remove(h)
                hit += 1
    if total:
        out.append(f"\nhonorific phrases reproduced exactly: {hit}/{total} ({hit / total:.1%})")
    return out


def load_model(opt, device):
    import torch
    from model import Model
    from utils import CTCLabelConverter

    converter = CTCLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    opt.device = device
    model = Model(opt).to(device)
    state = torch.load(opt.saved_model, map_location=device)
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    w = state.get("Prediction.weight")
    if w is not None and w.shape[0] != opt.num_class:
        raise SystemExit(f"{opt.saved_model} predicts {w.shape[0]} classes but --charset gives "
                         f"{opt.num_class}: use the charset the checkpoint was trained with.")
    model.load_state_dict(state)
    model.eval()
    return model, converter


def predict(model, converter, images, opt, device):
    """images: PIL images already flipped left-right, as the training data loader does."""
    import torch
    from dataset import AlignCollate
    collate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW)
    tensors, _ = collate([(im, "") for im in images])
    with torch.no_grad():
        preds = model(tensors.to(device))
    _, idx = preds.max(2)
    sizes = torch.IntTensor([preds.size(1)] * len(images))
    return converter.decode(idx.data, sizes.data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="image<TAB>label file")
    ap.add_argument("--root", default=None, help="folder the image paths are relative to (default: gt's folder)")
    ap.add_argument("--saved_model", required=True)
    ap.add_argument("--charset", "--glyphs", dest="charset", default="UrduGlyphs_extended.txt")
    ap.add_argument("--imgH", type=int, default=64)
    ap.add_argument("--imgW", type=int, default=800)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--FeatureExtraction", default="HRNet")
    ap.add_argument("--SequenceModeling", default="DBiLSTM")
    ap.add_argument("--hidden_size", type=int, default=256)
    ap.add_argument("--output_channel", type=int, default=512)
    ap.add_argument("--batch_max_length", type=int, default=250)
    ap.add_argument("--worst", type=int, default=15, help="how many worst lines to print")
    ap.add_argument("--out", help="write image/reference/prediction/CER for every line to this TSV")
    ap.add_argument("--device_id", default=None)
    opt = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    from urdu_text import load_charset

    opt.Prediction, opt.input_channel, opt.rgb = "CTC", 1, False
    if opt.FeatureExtraction == "HRNet":
        opt.output_channel = 32
    opt.character = load_charset(opt.charset)
    # the model keeps its temporal dropout on at test time (5 masks averaged): fix the seed so
    # two runs on the same checkpoint give the same numbers
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    device = torch.device(f"cuda:{opt.device_id}" if opt.device_id is not None else
                          "cuda" if torch.cuda.is_available() else "cpu")

    root = Path(opt.root) if opt.root else Path(opt.gt).parent
    items = []
    for line in Path(opt.gt).read_text(encoding="utf-8").split("\n"):
        if "\t" in line:
            path, label = line.split("\t", 1)
            items.append((path, clean(label)))
    if not items:
        raise SystemExit(f"no image<TAB>label lines in {opt.gt}")

    model, converter = load_model(opt, device)
    rows, squashed = [], 0
    for i in range(0, len(items), opt.batch_size):
        batch = items[i:i + opt.batch_size]
        images = []
        for path, _ in batch:
            im = Image.open(root / path).convert("L")
            squashed += im.width * opt.imgH / im.height > opt.imgW
            images.append(im.transpose(Image.FLIP_LEFT_RIGHT))
        preds = predict(model, converter, images, opt, device)
        rows += [(p, ref, clean(pred)) for (p, ref), pred in zip(batch, preds)]

    print(f"{len(rows)} lines, model {opt.saved_model}, input {opt.imgH}x{opt.imgW}")
    if squashed:
        print(f"note: {squashed} lines are wider than {opt.imgW}px at height {opt.imgH} and get squeezed "
              "horizontally (same as in training, but heavy squeezing costs accuracy)")
    print()
    print("\n".join(score(rows)))
    print("\nlenient CER: spaces ignored, look-alike letters ي/ی ى/ی ك/ک ة/ۃ counted as equal")

    ranked = sorted(rows, key=lambda r: edit_distance(r[1], r[2]) / max(1, len(r[1])), reverse=True)
    print(f"\nworst {min(opt.worst, len(ranked))} lines:")
    for path, ref, pred in ranked[:opt.worst]:
        print(f"  {path}  CER {edit_distance(ref, pred) / max(1, len(ref)):.1%}")
        print(f"    ref : {ref}")
        print(f"    pred: {pred}")

    if opt.out:
        with open(opt.out, "w", encoding="utf-8") as f:
            f.write("image\treference\tprediction\tcer\n")
            for path, ref, pred in rows:
                f.write(f"{path}\t{ref}\t{pred}\t{edit_distance(ref, pred) / max(1, len(ref)):.4f}\n")
        print(f"\nall lines written to {opt.out}")


if __name__ == "__main__":
    main()
