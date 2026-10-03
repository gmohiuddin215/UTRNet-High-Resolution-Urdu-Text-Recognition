"""
Page preprocessing for OCR: deskew a page, find its text lines and turn each line into the model's
input tensor. numpy only (PIL just to open files), no torch.

This is the reference implementation of ios/UrduOCR/Sources/UrduOCR/Preprocess.swift: every step
is written so the Swift port can repeat it operation for operation (integer arithmetic where
possible, explicit loops instead of library calls whose rounding differs between platforms).
Change one, change the other, and check with `ocr_page.py --dump` against `urdu-ocr --dump`.

Steps
  1. gray       8-bit luma with Pillow's formula; transparent areas count as white paper
  2. deskew     estimate the angle (-15..+15 deg) whose row projection of the ink is sharpest,
                then rotate the page by it (bilinear, white fill, canvas grows to fit)
  3. lines      same segmentation as page_to_lines.py, which cut the training data
  4. tensors    each line: trim side margins, mirror left-right, resize to height 64 with
                Pillow's exact bicubic filter, scale to [-1, 1], pad right to 800
"""
import math

import numpy as np

IMG_H, IMG_W = 64, 800
PAD, SIDE = 4, 8                    # px kept above/below and left/right of each line (as page_to_lines)
SKEW_LIMIT, SKEW_STEP = 300, 0.05   # angles are searched as k * 0.05 deg, |k| <= 300 (15 deg)


# ---------------------------------------------------------------------------- 1. gray

def load_gray(path_or_image):
    """uint8 [H, W], upright per EXIF. Alpha is composited onto white; RGB -> L uses Pillow's
    integer formula."""
    from PIL import Image, ImageOps
    im = path_or_image if isinstance(path_or_image, Image.Image) else Image.open(path_or_image)
    im = ImageOps.exif_transpose(im)                     # phone photos: turn upright as Swift does
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im.convert("RGBA"))
        im = bg
    if im.mode != "L":
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint32)
        luma = (rgb[..., 0] * 19595 + rgb[..., 1] * 38470 + rgb[..., 2] * 7471 + 0x8000) >> 16
        return luma.astype(np.uint8)
    return np.array(im, dtype=np.uint8)


# ---------------------------------------------------------------------------- shared helpers

def otsu(gray):
    """Threshold t such that ink = gray < t (same as page_to_lines.otsu, written as loops)."""
    hist = np.bincount(gray.ravel(), minlength=256)
    total = float(gray.size)
    omega, mu, om, m = [0.0] * 256, [0.0] * 256, 0.0, 0.0
    for i in range(256):
        p = hist[i] / total
        om += p
        m += p * i
        omega[i], mu[i] = om, m
    mu_t = mu[255]
    sigma = [0.0] * 256
    for i in range(256):
        denom = omega[i] * (1.0 - omega[i])
        if denom == 0.0:
            denom = 1e-9
        d = mu_t * omega[i] - mu[i]
        sigma[i] = d * d / denom
    best = max(sigma)
    tied = [i for i in range(256) if sigma[i] >= best - 1e-9]
    return (tied[0] + tied[-1]) // 2 + 1


def runs(mask):
    """(start, end) of consecutive True runs in a 1-D sequence."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def longest_run(row):
    best = cur = 0
    for v in row:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def row_runs(row):
    """(start, end) of ink runs in one row, vectorised (rows are long)."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], row.astype(np.int8), [0]))))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def percentile(values, q):
    """Linear-interpolated percentile of a list, spelled out so Swift gets the same number."""
    v = sorted(values)
    pos = q * (len(v) - 1)
    lo = int(math.floor(pos))
    if lo + 1 >= len(v):
        return float(v[-1])
    return float(v[lo]) + (float(v[lo + 1]) - float(v[lo])) * (pos - lo)


def median(values):
    v = sorted(values)
    n = len(v)
    return float(v[n // 2]) if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0


# ---------------------------------------------------------------------------- 2. deskew

def downscale(gray, f):
    """Mean of f x f blocks (integer floor), cropping the ragged edge."""
    if f <= 1:
        return gray
    h, w = gray.shape[0] // f, gray.shape[1] // f
    blocks = gray[:h * f, :w * f].reshape(h, f, w, f).astype(np.int64)
    return (blocks.sum(axis=(1, 3)) // (f * f)).astype(np.uint8)


def skew_score(xs, ys, k, size):
    """Sharpness of the row projection of ink points (centred coords) rotated by k * 0.05 deg."""
    a = k * SKEW_STEP * math.pi / 180.0
    s, c = math.sin(a), math.cos(a)
    rows = np.floor(-xs * s + ys * c + size).astype(np.int64)
    hist = np.bincount(rows, minlength=2 * size + 2)
    d = hist[1:] - hist[:-1]
    return int((d * d).sum())


def text_ink(ink, max_frac=0.25):
    """Ink without the components wider or taller than max_frac of the page: rules, frames and
    dark page edges are long and sharp, and would otherwise decide the angle instead of the text.
    Components are 8-connected runs joined with union-find, so Swift can do the same."""
    h, w = ink.shape
    row_runs_ = [row_runs(ink[y]) for y in range(h)]
    parent = []
    ids = []                                             # ids[y][i] -> run id
    for y in range(h):
        ids.append(list(range(len(parent), len(parent) + len(row_runs_[y]))))
        parent += ids[y]

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for y in range(1, h):
        prev, cur = row_runs_[y - 1], row_runs_[y]
        j = 0
        for i, (s, e) in enumerate(cur):
            while j < len(prev) and prev[j][1] < s:      # prev run ends left of this one
                j += 1
            jj = j
            while jj < len(prev) and prev[jj][0] <= e:    # overlaps or touches diagonally
                ra, rb = find(ids[y][i]), find(ids[y - 1][jj])
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
                jj += 1
    box = {}
    for y in range(h):
        for i, (s, e) in enumerate(row_runs_[y]):
            r = find(ids[y][i])
            b = box.get(r)
            box[r] = (s, e, y, y) if b is None else (min(b[0], s), max(b[1], e), b[2], y)
    out = ink.copy()
    for y in range(h):
        for i, (s, e) in enumerate(row_runs_[y]):
            x0, x1, y0, y1 = box[find(ids[y][i])]
            if x1 - x0 > max_frac * w or y1 - y0 + 1 > max_frac * h:
                out[y, s:e] = False
    return out


def estimate_skew(gray):
    """Angle step k (angle = k * 0.05 deg) that makes the text lines horizontal; 0 if no text."""
    f = max(1, max(gray.shape) // 1000)
    small = downscale(gray, f)
    ys, xs = np.nonzero(text_ink(small < otsu(small)))
    if len(xs) < 50:
        return 0
    h, w = small.shape
    xs = xs.astype(np.float64) + 0.5 - w / 2.0
    ys = ys.astype(np.float64) + 0.5 - h / 2.0
    size = int(math.ceil(math.sqrt(float(w * w + h * h)) / 2.0)) + 1

    def best_of(ks):
        best_k, best_s = ks[0], -1
        for k in ks:                                  # first maximum wins
            sc = skew_score(xs, ys, k, size)
            if sc > best_s:
                best_k, best_s = k, sc
        return best_k

    coarse = best_of(list(range(-SKEW_LIMIT, SKEW_LIMIT + 1, 10)))            # 0.5 deg steps
    fine = [k for k in range(coarse - 10, coarse + 11) if -SKEW_LIMIT <= k <= SKEW_LIMIT]
    return best_of(fine)                                                         # 0.05 deg steps


def rotate(gray, k):
    """Rotate by k * 0.05 deg (the same direction skew_score measures), bilinear, white fill."""
    if k == 0:
        return gray
    a = k * SKEW_STEP * math.pi / 180.0
    s, c = math.sin(a), math.cos(a)
    h, w = gray.shape
    nw = int(math.ceil(abs(w * c) + abs(h * s)))
    nh = int(math.ceil(abs(w * s) + abs(h * c)))
    qx = np.arange(nw, dtype=np.float64) + 0.5 - nw / 2.0
    qy = np.arange(nh, dtype=np.float64)[:, None] + 0.5 - nh / 2.0
    sx = qx * c - qy * s + (w / 2.0 - 0.5)
    sy = qx * s + qy * c + (h / 2.0 - 0.5)
    x0, y0 = np.floor(sx), np.floor(sy)
    fx, fy = sx - x0, sy - y0
    x0, y0 = x0.astype(np.int64), y0.astype(np.int64)
    src = gray.astype(np.float64)

    def px(yy, xx):
        inside = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        return np.where(inside, src[np.clip(yy, 0, h - 1), np.clip(xx, 0, w - 1)], 255.0)

    top = (1.0 - fx) * px(y0, x0) + fx * px(y0, x0 + 1)
    bot = (1.0 - fx) * px(y0 + 1, x0) + fx * px(y0 + 1, x0 + 1)
    v = (1.0 - fy) * top + fy * bot
    return np.clip(np.floor(v + 0.5), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------- 3. lines

def rule_rows(ink, min_frac):
    min_len = int(min_frac * ink.shape[1])
    counts = ink.sum(axis=1)
    return [y for y in range(ink.shape[0])
            if counts[y] >= min_len and any(e - s >= min_len for s, e in row_runs(ink[y]))]


def erase_rules(ink, min_frac, margin=3):
    out = ink.copy()
    min_len = int(min_frac * ink.shape[1])
    counts = ink.sum(axis=1)
    for y in range(ink.shape[0]):
        if counts[y] < min_len:
            continue
        for s, e in row_runs(ink[y]):
            if e - s >= min_len:
                out[max(0, y - margin):y + margin + 1, s:e] = False
    return out


def is_rule(band, min_frac=0.05, share=0.7):
    total = int(band.sum())
    if not total:
        return True
    min_len = max(20, int(min_frac * band.shape[1]))
    in_runs = sum(e - s for row in band for s, e in row_runs(row) if e - s >= min_len)
    return in_runs > share * total


def split_tall(lines, smooth, tall=1.6, part=0.5, dip=0.4):
    if len(lines) < 3:
        return lines
    typical = median([b - t for t, b in lines])
    reach, gap = int(typical), int(part * typical)
    out = []
    for t, b in lines:
        if b - t <= tall * typical:
            out.append((t, b))
            continue
        valleys = []
        for y in range(t + gap, b - gap):
            left = smooth[max(t, y - reach):y]
            right = smooth[y + 1:min(b, y + 1 + reach)]
            window = smooth[max(t, y - gap // 2):y + gap // 2 + 1]
            if (len(left) and len(right) and smooth[y] == min(window)
                    and smooth[y] <= dip * min(max(left), max(right))):
                valleys.append(y)
        cuts = []
        for y in sorted(valleys, key=lambda v: (smooth[v], v)):     # deepest first, keep apart
            if all(abs(y - c) >= gap for c in cuts):
                cuts.append(y)
        edges = [t] + sorted(cuts) + [b]
        out += list(zip(edges, edges[1:]))
    return out


def find_lines(gray, split=0.12, low=0.02, strong=0.5, rule=0.15):
    """(top, bottom, left, right) of every text line, top to bottom, padded like the training crops."""
    h, w = gray.shape
    raw = gray < otsu(gray)
    ink = erase_rules(raw, rule)
    min_gap, min_height = max(4, h // 300), max(12, h // 120)

    proj = ink.sum(axis=1).astype(np.int64).tolist()
    k = max(3, h // 400) | 1
    half = k // 2
    smooth = []
    for y in range(h):                                   # centred moving mean, integer sum first
        smooth.append(sum(proj[max(0, y - half):min(h, y + half + 1)]) / k)
    positive = [v for v in smooth if v > 0]
    if not positive:
        return []
    ref = percentile(positive, 0.9)
    for y in rule_rows(raw, rule):                       # a rule always separates lines
        for yy in range(max(0, y - 1), min(h, y + 2)):
            smooth[yy] = 0.0

    regions = []
    for r in runs([v > max(1.0, low * ref) for v in smooth]):
        if regions and r[0] - regions[-1][1] < min_gap:
            regions[-1] = (regions[-1][0], r[1])
        else:
            regions.append(r)

    lines = []
    for top, bottom in regions:
        if bottom - top < min_height:
            continue
        seg = smooth[top:bottom]
        cores = [c for c in runs([v > split * ref for v in seg])
                 if c[1] - c[0] >= max(3, min_height // 4) and max(seg[c[0]:c[1]]) >= strong * ref]
        if len(cores) <= 1:
            lines.append((top, bottom))
            continue
        cuts = [top]
        for (_, e1), (s2, _) in zip(cores, cores[1:]):
            gapseg = seg[e1:s2]
            cuts.append(top + e1 + gapseg.index(min(gapseg)))
        cuts.append(bottom)
        lines += [(a, b) for a, b in zip(cuts, cuts[1:]) if b - a >= min_height]
    lines = [l for l in split_tall(lines, smooth) if not is_rule(raw[l[0]:l[1]])]

    boxes = []
    for t, b in lines:
        t, b = max(0, t - PAD), min(h, b + PAD)
        cols = np.flatnonzero(ink[t:b].any(axis=0))
        if not len(cols):
            continue
        boxes.append((t, b, max(0, int(cols[0]) - SIDE), min(w, int(cols[-1]) + SIDE)))
    return boxes


# ---------------------------------------------------------------------------- 4. tensors

def _bicubic(x):
    a = -0.5
    if x < 0.0:
        x = -x
    if x < 1.0:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1
    if x < 2.0:
        return (((x - 5) * x + 8) * x - 4) * a
    return 0.0


def _coeffs(in_size, out_size):
    """Pillow's precompute_coeffs + normalize_coeffs_8bpc for the bicubic filter."""
    scale = in_size / out_size
    filterscale = max(scale, 1.0)
    support = 2.0 * filterscale
    ksize = int(math.ceil(support)) * 2 + 1
    bounds, kk = [], []
    for xx in range(out_size):
        center = (xx + 0.5) * scale
        ss = 1.0 / filterscale
        xmin = max(0, int(center - support + 0.5))      # int() truncates toward zero, like C
        xmax = min(in_size, int(center + support + 0.5)) - xmin
        k = [_bicubic((x + xmin - center + 0.5) * ss) for x in range(xmax)]
        ww = 0.0
        for v in k:
            ww += v
        if ww != 0.0:
            k = [v / ww for v in k]
        ik = [int(-0.5 + v * (1 << 22)) if v < 0 else int(0.5 + v * (1 << 22)) for v in k]
        kk.append(ik + [0] * (ksize - len(ik)))
        bounds.append((xmin, xmax))
    return bounds, kk, ksize


def _pass(img, out_size):
    """One 8-bit resampling pass along the last axis (Pillow's ImagingResampleHorizontal_8bpc)."""
    bounds, kk, ksize = _coeffs(img.shape[1], out_size)
    idx = np.array([[min(b[0] + i, img.shape[1] - 1) for i in range(ksize)] for b in bounds])
    weights = np.array(kk, dtype=np.int64)
    acc = (img.astype(np.int64)[:, idx] * weights[None]).sum(axis=2) + (1 << 21)
    return np.clip(acc >> 22, 0, 255).astype(np.uint8)


def resize_bicubic(img, out_w, out_h):
    """Bit-exact Image.resize((out_w, out_h), Image.BICUBIC) for mode L."""
    if out_w != img.shape[1]:
        img = _pass(img, out_w)
    if out_h != img.shape[0]:
        img = _pass(img.T, out_h).T
    return np.ascontiguousarray(img)


def line_input(gray, box):
    """Model input [1, 64, 800] float32 for one line box, plus the resized uint8 line."""
    t, b, l, r = box
    return line_tensor(gray[t:b, l:r])


def line_tensor(line):
    """Model input for a cropped line image (uint8 [h, w]), exactly as train.py's AlignCollate
    prepares it: mirrored, bicubic to height 64, [-1, 1], right edge repeated to width 800."""
    crop = line[:, ::-1]                                 # the model reads mirrored lines
    h, w = crop.shape
    rw = min(IMG_W, int(math.ceil(IMG_H * (w / float(h)))))
    small = resize_bicubic(crop, rw, IMG_H)
    x = (small.astype(np.float32) / np.float32(255.0) - np.float32(0.5)) / np.float32(0.5)
    out = np.empty((1, IMG_H, IMG_W), dtype=np.float32)
    out[0, :, :rw] = x
    out[0, :, rw:] = x[:, rw - 1:rw]                    # repeat the last column, as NormalizePAD
    return out, small


def prepare_page(gray, deskew=True):
    """-> (angle step k, rotated page, [(box, tensor, resized uint8 line), ...])"""
    k = estimate_skew(gray) if deskew else 0
    page = rotate(gray, k)
    lines = []
    for box in find_lines(page):
        x, small = line_input(page, box)
        lines.append((box, x, small))
    return k, page, lines


# ---------------------------------------------------------------------------- decoding

def ctc_decode(indices, charset):
    """Greedy CTC: drop repeats, then blanks (index 0). charset[i - 1] is class i."""
    out, prev = [], 0
    for i in indices:
        i = int(i)
        if i != 0 and i != prev:
            out.append(charset[i - 1])
        prev = i
    return "".join(out)
