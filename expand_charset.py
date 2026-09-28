#!/usr/bin/env python3
"""
Expand a UTRNet checkpoint's CTC output layer so it can predict the added characters.

The pretrained UTRNet-Large knows 180 glyphs + space. UrduGlyphs_extended.txt adds 29
(alef maksura, alef with hamza below, dammatan, alef wasla, the missing Arabic-Indic digits,
Quranic pause marks, ornate parentheses ...), appended after the original list.
Only the final Linear layer changes size: its rows are copied across by *character*, so every
glyph the model already knows keeps its learned weights, and only the new rows start fresh.
Everything else - the HRNet feature extractor and both BiLSTMs - is untouched.

Run from inside the UTRNet repo folder:

  python expand_charset.py --ckpt UTRNet-Large.pth --out UTRNet-Large-extended.pth

Then pass --charset UrduGlyphs_extended.txt to train.py / test.py / read.py / eval_lines.py.
CTC checkpoints only (the released UTRNet models are CTC).
"""
import argparse, torch


def charset(path):
    """Same parsing as read.py/train.py: one glyph per line, then a trailing space char."""
    from urdu_text import load_charset
    return list(load_charset(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_glyphs", default="UrduGlyphs.txt")
    ap.add_argument("--new_glyphs", default="UrduGlyphs_extended.txt")
    ap.add_argument("--ckpt", required=True, help="pretrained UTRNet .pth")
    ap.add_argument("--out", required=True, help="where to write the expanded checkpoint")
    ap.add_argument("--layer", default="Prediction", help="name of the final Linear layer")
    a = ap.parse_args()

    old, new = charset(a.old_glyphs), charset(a.new_glyphs)
    print(f"old charset {len(old)} chars (+1 CTC blank) -> new {len(new)}")
    added = [c for c in new if c not in old]
    dropped = [c for c in old if c not in new]
    print("added:", "".join(added) or "(none)")
    if dropped:
        raise SystemExit(f"Refusing to run: the new list drops {len(dropped)} existing glyph(s): "
                         + "".join(dropped) + "\nOnly add glyphs; never remove them.")

    sd = torch.load(a.ckpt, map_location="cpu")
    sd = sd.get("state_dict", sd)
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}   # saved with DataParallel

    wk, bk = f"{a.layer}.weight", f"{a.layer}.bias"
    if wk not in sd or bk not in sd:
        raise SystemExit(f"No '{a.layer}' weight found. Keys look like: "
                         + ", ".join(list(sd)[-5:]))

    w_old, b_old = sd[wk], sd[bk]
    n_old, hidden = w_old.shape
    if n_old != len(old) + 1:
        raise SystemExit(f"{wk} has {n_old} rows but the old charset implies {len(old)+1}. "
                         "Check that --old_glyphs matches the file this checkpoint was trained with.")

    n_new = len(new) + 1
    w = torch.zeros(n_new, hidden, dtype=w_old.dtype)
    b = torch.zeros(n_new, dtype=b_old.dtype)
    torch.nn.init.normal_(w, std=0.01)           # fresh rows start small
    b.fill_(float(b_old.mean()))

    w[0], b[0] = w_old[0], b_old[0]              # CTC blank keeps index 0
    idx_old = {c: i + 1 for i, c in enumerate(old)}
    copied = 0
    for i, c in enumerate(new):
        if c in idx_old:
            w[i + 1], b[i + 1] = w_old[idx_old[c]], b_old[idx_old[c]]
            copied += 1

    sd[wk], sd[bk] = w, b
    torch.save(sd, a.out)
    print(f"copied {copied}/{n_new} rows (blank + {copied - 1} known glyphs), "
          f"{n_new - copied} new rows initialised")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
