#!/usr/bin/env python3
"""
Convert the trained UTRNet checkpoint to Core ML for the iPhone/iPad/Mac app, and prove the
conversion lost nothing.

  pip install torch==2.7.0 coremltools==9.0 numpy pillow pymupdf
  python export_coreml.py --saved_model results/utrnet_hadith_final.pth \\
      --charset results/UrduGlyphs_extended.txt --out UrduOCR.mlpackage \\
      --gt real_val/gt.txt --gt khatme/val/gt.txt

The package holds the network in full 32-bit precision (no fp16, no quantisation) with the 5
temporal-dropout masks frozen inside (see deploy_model.py), and the charset in its metadata, so
the app needs only this one file.

--batch N (default 8) makes the package read N lines per call. The LSTM half of the network
has to step through each line's 800 columns one after another, so one line at a time leaves the
CPU/GPU mostly idle; with N lines every step does N lines of work for about the cost of one.
The arithmetic per line is the same, and the verification below checks the batched package
against PyTorch reading each line on its own. ocr_page.py and the Swift package read the batch
size from the package and pad the last batch, so --batch 1 still works with them.

Verification (macOS only; Core ML does not run elsewhere) runs every --gt line image, and every
--page image, through
  * PyTorch, the deterministic model (the reference)
  * Core ML on the CPU, and Core ML on CPU+GPU
and reports the largest logit difference, how many of the 800 per-line argmax decisions match,
how many decoded lines are identical, and CER against the labels. "identical text 252/252" is
the claim that matters: the app reads exactly what PyTorch reads.

--fp16 additionally writes a half-precision package (half the size, can use the Neural Engine)
and verifies it the same way, so you can see what it would cost before choosing it.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np


def convert(deploy, charset, out, fp16=False, batch=1):
    import coremltools as ct
    import torch
    example = torch.zeros(batch, 1, 64, 800)
    traced = torch.jit.trace(deploy, example)
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=(batch, 1, 64, 800), dtype=np.float32)],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS16)
    ml.short_description = (f"UTRNet fine-tuned for Urdu hadith books. Input: {batch} text lines, "
                            "each mirrored, height 64, padded to 800, scaled to [-1, 1]. Output: CTC "
                            f"logits [{batch}, 800, classes], class 0 = blank.")
    ml.user_defined_metadata["charset"] = json.dumps(list(charset), ensure_ascii=False)
    ml.user_defined_metadata["precision"] = "float16" if fp16 else "float32"
    ml.user_defined_metadata["input"] = f"{batch}x1x64x800"
    ml.save(str(out))
    return out


def gt_lines(gt_files, limit):
    from ocr_pipeline import load_gray
    from urdu_text import clean
    items = []
    for gt in gt_files:
        root = Path(gt).parent
        for row in Path(gt).read_text(encoding="utf-8").split("\n"):
            if "\t" in row:
                path, label = row.split("\t", 1)
                items.append((f"{gt}:{path}", load_gray(root / path), clean(label)))
    return items[:limit] if limit else items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saved_model", required=True, help="the trained .pth")
    ap.add_argument("--charset", required=True, help="the charset it was trained with")
    ap.add_argument("--out", default="UrduOCR.mlpackage")
    ap.add_argument("--batch", type=int, default=8, help="lines read per call (1 = one line at a time)")
    ap.add_argument("--fp16", action="store_true", help="also export and verify a float16 package")
    ap.add_argument("--gt", action="append", default=[], help="image<TAB>label file to verify on (repeatable)")
    ap.add_argument("--page", action="append", default=[], help="page image to verify on, whole pipeline (repeatable)")
    ap.add_argument("--limit", type=int, default=0, help="verify on the first N labeled lines only")
    ap.add_argument("--skip_verify", action="store_true")
    a = ap.parse_args()

    import torch
    from deploy_model import load_charset, load_deploy, load_trained
    from ocr_pipeline import ctc_decode, line_tensor, prepare_page, load_gray
    from urdu_text import clean
    from eval_lines import edit_distance, lenient

    torch.set_grad_enabled(False)
    charset = load_charset(a.charset)
    deploy = load_deploy(a.saved_model, charset)

    outs = [Path(a.out)]
    t = time.time()
    convert(deploy, charset, outs[0], batch=a.batch)
    print(f"wrote {outs[0]} (float32, {a.batch} lines per call) in {time.time() - t:.0f}s")
    if a.fp16:
        outs.append(outs[0].with_name(outs[0].stem + "_fp16.mlpackage"))
        convert(deploy, charset, outs[1], fp16=True, batch=a.batch)
        print(f"wrote {outs[1]} (float16)")

    if a.skip_verify:
        return
    if platform.system() != "Darwin":
        print("\nverification needs macOS (Core ML only runs on Apple systems): run this on your Mac")
        return
    if not a.gt and not a.page:
        print("\nnothing to verify on: pass --gt real_val/gt.txt (and/or --page some_page.png)")
        return

    # inputs: labeled line images, then every line of every --page through the full pipeline
    inputs = [(name, line_tensor(gray)[0], label) for name, gray, label in gt_lines(a.gt, a.limit)]
    for p in a.page:
        _, _, lines = prepare_page(load_gray(p))
        inputs += [(f"{p}:line{i + 1}", x, None) for i, (_, x, _) in enumerate(lines)]
    print(f"\nverifying on {len(inputs)} lines")

    import coremltools as ct
    original = load_trained(a.saved_model, charset)       # model.py as trained, random masks
    np.random.seed(0)
    ref_logits, ref_text, orig_text = [], [], []
    t = time.time()
    for _, x, _ in inputs:
        xt = torch.from_numpy(x[None])
        lg = deploy(xt)[0].numpy()
        ref_logits.append(lg)
        ref_text.append(ctc_decode(lg.argmax(1), charset))
        orig_text.append(ctc_decode(original(xt)[0].argmax(1).numpy(), charset))
    print(f"PyTorch: {time.time() - t:.0f}s")

    labels = [(i, lab) for i, (_, _, lab) in enumerate(inputs) if lab is not None]

    def cer(texts, loose=False):
        f = lenient if loose else (lambda s: s)
        errs = sum(edit_distance(f(lab), f(clean(texts[i]))) for i, lab in labels)
        return errs / max(1, sum(len(f(lab)) for _, lab in labels))

    rows = [("PyTorch, model.py (random dropout masks)", orig_text, None, None),
            ("PyTorch, deterministic (reference)", ref_text, None, None)]
    for pkg in outs:
        for units_name, units in (("CPU", ct.ComputeUnit.CPU_ONLY), ("CPU+GPU", ct.ComputeUnit.CPU_AND_GPU),
                                  ("all units", ct.ComputeUnit.ALL)):
            if units_name == "all units" and pkg == outs[0]:
                continue                          # float32 cannot use the Neural Engine anyway
            ml = ct.models.MLModel(str(pkg), compute_units=units)
            ml.predict({"image": np.zeros((a.batch, 1, 64, 800), np.float32)})   # warm-up, not timed
            texts, max_diff, same_argmax = [], 0.0, 0
            t = time.time()
            for s in range(0, len(inputs), a.batch):
                xs = [x for _, x, _ in inputs[s:s + a.batch]]
                n = len(xs)
                xs += [xs[-1]] * (a.batch - n)                       # pad the last batch
                out = ml.predict({"image": np.stack(xs)})["logits"]
                for lg, ref in zip(out[:n], ref_logits[s:s + n]):
                    max_diff = max(max_diff, float(np.abs(lg - ref).max()))
                    same_argmax += int((lg.argmax(1) == ref.argmax(1)).sum())
                    texts.append(ctc_decode(lg.argmax(1), charset))
            dt = time.time() - t
            print(f"Core ML {pkg.name} on {units_name}: {dt:.1f}s, {dt / len(inputs):.2f}s per line")
            rows.append((f"Core ML {pkg.name}, {units_name}", texts, max_diff,
                         same_argmax / (len(inputs) * 800)))

    print(f"\n{'model':58} {'identical text':>15} {'max |logit diff|':>17} {'argmax match':>13}"
          + (f" {'CER':>7} {'lenient':>8}" if labels else ""))
    for name, texts, diff, argm in rows:
        same = sum(x == y for x, y in zip(texts, ref_text))
        line = (f"{name:58} {same:>7}/{len(inputs):<7} "
                f"{'-' if diff is None else f'{diff:.2e}':>17} {'-' if argm is None else f'{argm:.4%}':>13}")
        if labels:
            line += f" {cer(texts):7.2%} {cer(texts, True):8.2%}"
        print(line)

    for name, texts, _, _ in rows[2:]:
        diffs = [i for i, (x, y) in enumerate(zip(texts, ref_text)) if x != y]
        for i in diffs[:5]:
            print(f"\n{name} differs on {inputs[i][0]}\n  torch : {ref_text[i]}\n  coreml: {texts[i]}")
    print("\n'identical text' compares each line with the deterministic PyTorch model reading that line "
          "on its own. The first row "
          "is model.py as it ran during training, with fresh random masks; it is expected to differ "
          "on a few lines, which is why the exported model freezes the masks.")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
