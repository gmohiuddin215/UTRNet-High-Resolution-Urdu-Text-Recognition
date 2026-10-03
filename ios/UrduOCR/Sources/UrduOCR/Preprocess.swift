import Foundation

// Swift port of ocr_pipeline.py: deskew, line finding and model input, step for step.
// The Python file is the reference; keep the two in sync and compare them with
// `ocr_page.py --dump` and `urdu-ocr --dump` on the same PNG page.

/// A text line in the deskewed page: rows top..<bottom, columns left..<right.
public struct LineBox: Equatable {
    public let top: Int, bottom: Int, left: Int, right: Int
}

public struct PreparedLine {
    public let box: LineBox
    /// Model input, 1 x 64 x 800 row-major, in [-1, 1].
    public let input: [Float]
    /// The mirrored line at height 64 before scaling (for checksums and debugging).
    public let resized: GrayImage
}

public struct PreparedPage {
    /// Deskew angle in steps of 0.05 degrees.
    public let skewStep: Int
    public var skewDegrees: Double { Double(skewStep) * Preprocess.skewStep }
    /// The page after deskewing (line boxes refer to it).
    public let page: GrayImage
    public let lines: [PreparedLine]
}

/// Binary image, row-major.
struct Mask {
    let width: Int, height: Int
    var bits: [Bool]

    func rowCount(_ y: Int) -> Int {
        var n = 0
        for x in 0..<width where bits[y * width + x] { n += 1 }
        return n
    }

    /// (start, end) of the ink runs in row y.
    func rowRuns(_ y: Int) -> [(Int, Int)] {
        var out: [(Int, Int)] = []
        var start = -1
        let base = y * width
        for x in 0..<width {
            if bits[base + x] {
                if start < 0 { start = x }
            } else if start >= 0 {
                out.append((start, x))
                start = -1
            }
        }
        if start >= 0 { out.append((start, width)) }
        return out
    }
}

public enum Preprocess {
    public static let imgH = 64, imgW = 800
    static let pad = 4, side = 8
    static let skewLimit = 300            // angles are k * 0.05 deg, |k| <= 300 (15 deg)
    public static let skewStep = 0.05

    /// Deskew the page, find its lines, and build each line's model input.
    public static func preparePage(_ gray: GrayImage, deskew: Bool = true) -> PreparedPage {
        let k = deskew ? estimateSkew(gray) : 0
        let page = rotate(gray, k)
        let lines = findLines(page).map { box -> PreparedLine in
            let (x, small) = lineInput(page, box)
            return PreparedLine(box: box, input: x, resized: small)
        }
        return PreparedPage(skewStep: k, page: page, lines: lines)
    }

    // MARK: - shared helpers

    static func otsu(_ pixels: [UInt8]) -> Int {
        var hist = [Int](repeating: 0, count: 256)
        for p in pixels { hist[Int(p)] += 1 }
        let total = Double(pixels.count)
        var omega = [Double](repeating: 0, count: 256), mu = [Double](repeating: 0, count: 256)
        var om = 0.0, m = 0.0
        for i in 0..<256 {
            let p = Double(hist[i]) / total
            om += p
            m += p * Double(i)
            omega[i] = om
            mu[i] = m
        }
        let muT = mu[255]
        var sigma = [Double](repeating: 0, count: 256)
        for i in 0..<256 {
            var denom = omega[i] * (1.0 - omega[i])
            if denom == 0.0 { denom = 1e-9 }
            let d = muT * omega[i] - mu[i]
            sigma[i] = d * d / denom
        }
        let best = sigma.max()!
        let tied = (0..<256).filter { sigma[$0] >= best - 1e-9 }
        return (tied.first! + tied.last!) / 2 + 1
    }

    static func runs(_ mask: [Bool]) -> [(Int, Int)] {
        var out: [(Int, Int)] = []
        var start = -1
        for (i, v) in mask.enumerated() {
            if v && start < 0 {
                start = i
            } else if !v && start >= 0 {
                out.append((start, i))
                start = -1
            }
        }
        if start >= 0 { out.append((start, mask.count)) }
        return out
    }

    static func percentile(_ values: [Double], _ q: Double) -> Double {
        let v = values.sorted()
        let pos = q * Double(v.count - 1)
        let lo = Int(floor(pos))
        if lo + 1 >= v.count { return v[v.count - 1] }
        return v[lo] + (v[lo + 1] - v[lo]) * (pos - Double(lo))
    }

    static func median(_ values: [Int]) -> Double {
        let v = values.sorted(), n = v.count
        return n % 2 == 1 ? Double(v[n / 2]) : Double(v[n / 2 - 1] + v[n / 2]) / 2.0
    }

    /// Python-style slice a[lo:hi] (clipped, empty when lo >= hi).
    static func slice(_ a: [Double], _ lo: Int, _ hi: Int) -> ArraySlice<Double> {
        let l = max(0, min(lo, a.count))
        let u = max(l, min(hi, a.count))
        return a[l..<u]
    }

    static func threshold(_ g: GrayImage) -> Mask {
        let t = otsu(g.pixels)
        return Mask(width: g.width, height: g.height, bits: g.pixels.map { Int($0) < t })
    }

    // MARK: - deskew

    static func downscale(_ g: GrayImage, _ f: Int) -> GrayImage {
        if f <= 1 { return g }
        let h = g.height / f, w = g.width / f
        var px = [UInt8](repeating: 0, count: w * h)
        for y in 0..<h {
            for x in 0..<w {
                var s = 0
                for dy in 0..<f {
                    let row = (y * f + dy) * g.width + x * f
                    for dx in 0..<f { s += Int(g.pixels[row + dx]) }
                }
                px[y * w + x] = UInt8(s / (f * f))
            }
        }
        return GrayImage(width: w, height: h, pixels: px)
    }

    /// Ink without components wider or taller than maxFrac of the page (rules, frames, page edges).
    static func textInk(_ ink: Mask, maxFrac: Double = 0.25) -> Mask {
        let h = ink.height, w = ink.width
        var rr: [[(Int, Int)]] = []
        var ids: [[Int]] = []
        var parent: [Int] = []
        for y in 0..<h {
            let r = ink.rowRuns(y)
            rr.append(r)
            ids.append(Array(parent.count..<(parent.count + r.count)))
            parent.append(contentsOf: ids[y])
        }
        func find(_ start: Int) -> Int {
            var a = start
            while parent[a] != a {
                parent[a] = parent[parent[a]]
                a = parent[a]
            }
            return a
        }
        if h > 1 {
            for y in 1..<h {
                let prev = rr[y - 1], cur = rr[y]
                var j = 0
                for (i, run) in cur.enumerated() {
                    let (s, e) = run
                    while j < prev.count && prev[j].1 < s { j += 1 }
                    var jj = j
                    while jj < prev.count && prev[jj].0 <= e {
                        let ra = find(ids[y][i]), rb = find(ids[y - 1][jj])
                        if ra != rb { parent[max(ra, rb)] = min(ra, rb) }
                        jj += 1
                    }
                }
            }
        }
        var box: [Int: (Int, Int, Int, Int)] = [:]
        for y in 0..<h {
            for (i, run) in rr[y].enumerated() {
                let r = find(ids[y][i])
                if let b = box[r] {
                    box[r] = (min(b.0, run.0), max(b.1, run.1), b.2, y)
                } else {
                    box[r] = (run.0, run.1, y, y)
                }
            }
        }
        var out = ink
        for y in 0..<h {
            for (i, run) in rr[y].enumerated() {
                let b = box[find(ids[y][i])]!
                if Double(b.1 - b.0) > maxFrac * Double(w) || Double(b.3 - b.2 + 1) > maxFrac * Double(h) {
                    for x in run.0..<run.1 { out.bits[y * w + x] = false }
                }
            }
        }
        return out
    }

    /// Angle step k (angle = k * 0.05 deg) that makes the text lines horizontal; 0 if no text.
    public static func estimateSkew(_ gray: GrayImage) -> Int {
        let f = max(1, max(gray.width, gray.height) / 1000)
        let small = downscale(gray, f)
        let ink = textInk(threshold(small))
        let w = small.width, h = small.height
        var xs: [Double] = [], ys: [Double] = []
        for y in 0..<h {
            for x in 0..<w where ink.bits[y * w + x] {
                xs.append(Double(x) + 0.5 - Double(w) / 2.0)
                ys.append(Double(y) + 0.5 - Double(h) / 2.0)
            }
        }
        if xs.count < 50 { return 0 }
        let size = Int(ceil(sqrt(Double(w * w + h * h)) / 2.0)) + 1

        func score(_ k: Int) -> Int {
            let a = Double(k) * skewStep * Double.pi / 180.0
            let s = sin(a), c = cos(a)
            var hist = [Int](repeating: 0, count: 2 * size + 2)
            let sz = Double(size)
            for i in 0..<xs.count {
                let row = -xs[i] * s + ys[i] * c + sz
                hist[Int(floor(row))] += 1
            }
            var total = 0
            for i in 1..<hist.count {
                let d = hist[i] - hist[i - 1]
                total += d * d
            }
            return total
        }
        func bestOf(_ ks: [Int]) -> Int {
            var bestK = ks[0], bestS = -1
            for k in ks {
                let sc = score(k)
                if sc > bestS {
                    bestK = k
                    bestS = sc
                }
            }
            return bestK
        }
        let coarse = bestOf(Array(stride(from: -skewLimit, through: skewLimit, by: 10)))
        let fine = (coarse - 10...coarse + 10).filter { $0 >= -skewLimit && $0 <= skewLimit }
        return bestOf(fine)
    }

    /// Rotate by k * 0.05 deg (the direction estimateSkew measures), bilinear, white fill.
    public static func rotate(_ g: GrayImage, _ k: Int) -> GrayImage {
        if k == 0 { return g }
        let a = Double(k) * skewStep * Double.pi / 180.0
        let s = sin(a), c = cos(a)
        let w = g.width, h = g.height
        let nw = Int(ceil(abs(Double(w) * c) + abs(Double(h) * s)))
        let nh = Int(ceil(abs(Double(w) * s) + abs(Double(h) * c)))
        let cx = Double(w) / 2.0 - 0.5, cy = Double(h) / 2.0 - 0.5
        var out = [UInt8](repeating: 255, count: nw * nh)
        g.pixels.withUnsafeBufferPointer { src in
            @inline(__always) func px(_ yy: Int, _ xx: Int) -> Double {
                (xx >= 0 && xx < w && yy >= 0 && yy < h) ? Double(src[yy * w + xx]) : 255.0
            }
            for Y in 0..<nh {
                let qy = Double(Y) + 0.5 - Double(nh) / 2.0
                for X in 0..<nw {
                    let qx = Double(X) + 0.5 - Double(nw) / 2.0
                    let sx = qx * c - qy * s + cx
                    let sy = qx * s + qy * c + cy
                    let fx0 = floor(sx), fy0 = floor(sy)
                    let fx = sx - fx0, fy = sy - fy0
                    let x0 = Int(fx0), y0 = Int(fy0)
                    let top = (1.0 - fx) * px(y0, x0) + fx * px(y0, x0 + 1)
                    let bot = (1.0 - fx) * px(y0 + 1, x0) + fx * px(y0 + 1, x0 + 1)
                    let v = (1.0 - fy) * top + fy * bot
                    out[Y * nw + X] = UInt8(min(255.0, max(0.0, floor(v + 0.5))))
                }
            }
        }
        return GrayImage(width: nw, height: nh, pixels: out)
    }

    // MARK: - lines

    static func ruleRows(_ ink: Mask, _ minFrac: Double) -> [Int] {
        let minLen = Int(minFrac * Double(ink.width))
        var rows: [Int] = []
        for y in 0..<ink.height where ink.rowCount(y) >= minLen {
            if ink.rowRuns(y).contains(where: { $0.1 - $0.0 >= minLen }) { rows.append(y) }
        }
        return rows
    }

    static func eraseRules(_ ink: Mask, _ minFrac: Double, margin: Int = 3) -> Mask {
        var out = ink
        let minLen = Int(minFrac * Double(ink.width))
        for y in 0..<ink.height where ink.rowCount(y) >= minLen {
            for (s, e) in ink.rowRuns(y) where e - s >= minLen {
                for yy in max(0, y - margin)..<min(ink.height, y + margin + 1) {
                    for x in s..<e { out.bits[yy * ink.width + x] = false }
                }
            }
        }
        return out
    }

    static func isRule(_ ink: Mask, _ top: Int, _ bottom: Int, minFrac: Double = 0.05, share: Double = 0.7) -> Bool {
        var total = 0
        for y in top..<bottom { total += ink.rowCount(y) }
        if total == 0 { return true }
        let minLen = max(20, Int(minFrac * Double(ink.width)))
        var inRuns = 0
        for y in top..<bottom {
            for (s, e) in ink.rowRuns(y) where e - s >= minLen { inRuns += e - s }
        }
        return Double(inRuns) > share * Double(total)
    }

    static func splitTall(_ lines: [(Int, Int)], _ smooth: [Double],
                          tall: Double = 1.6, part: Double = 0.5, dip: Double = 0.4) -> [(Int, Int)] {
        if lines.count < 3 { return lines }
        let typical = median(lines.map { $0.1 - $0.0 })
        let reach = Int(typical), gap = Int(part * typical)
        var out: [(Int, Int)] = []
        for (t, b) in lines {
            if Double(b - t) <= tall * typical {
                out.append((t, b))
                continue
            }
            var valleys: [Int] = []
            if t + gap < b - gap {
                for y in (t + gap)..<(b - gap) {
                    let left = slice(smooth, max(t, y - reach), y)
                    let right = slice(smooth, y + 1, min(b, y + 1 + reach))
                    let window = slice(smooth, max(t, y - gap / 2), y + gap / 2 + 1)
                    if !left.isEmpty && !right.isEmpty && smooth[y] == window.min()!
                        && smooth[y] <= dip * min(left.max()!, right.max()!) {
                        valleys.append(y)
                    }
                }
            }
            let order = valleys.sorted { (smooth[$0], $0) < (smooth[$1], $1) }   // deepest first
            var cuts: [Int] = []
            for y in order where cuts.allSatisfy({ abs(y - $0) >= gap }) { cuts.append(y) }
            let edges = [t] + cuts.sorted() + [b]
            for i in 0..<(edges.count - 1) { out.append((edges[i], edges[i + 1])) }
        }
        return out
    }

    /// Text lines of a deskewed page, top to bottom, padded like the training crops.
    public static func findLines(_ g: GrayImage, split: Double = 0.12, low: Double = 0.02,
                                 strong: Double = 0.5, rule: Double = 0.15) -> [LineBox] {
        let h = g.height, w = g.width
        let raw = threshold(g)
        let ink = eraseRules(raw, rule)
        let minGap = max(4, h / 300), minHeight = max(12, h / 120)

        var proj = [Int](repeating: 0, count: h)
        for y in 0..<h { proj[y] = ink.rowCount(y) }
        let k = max(3, h / 400) | 1
        let half = k / 2
        var smooth = [Double](repeating: 0, count: h)
        for y in 0..<h {                            // centred moving mean, integer sum first
            var s = 0
            for yy in max(0, y - half)..<min(h, y + half + 1) { s += proj[yy] }
            smooth[y] = Double(s) / Double(k)
        }
        let positive = smooth.filter { $0 > 0 }
        if positive.isEmpty { return [] }
        let ref = percentile(positive, 0.9)
        for y in ruleRows(raw, rule) {              // a rule always separates lines
            for yy in max(0, y - 1)..<min(h, y + 2) { smooth[yy] = 0 }
        }

        var regions: [(Int, Int)] = []
        for r in runs(smooth.map { $0 > max(1.0, low * ref) }) {
            if let last = regions.last, r.0 - last.1 < minGap {
                regions[regions.count - 1] = (last.0, r.1)
            } else {
                regions.append(r)
            }
        }

        var lines: [(Int, Int)] = []
        for (top, bottom) in regions {
            if bottom - top < minHeight { continue }
            let seg = Array(smooth[top..<bottom])
            let cores = runs(seg.map { $0 > split * ref }).filter { c in
                c.1 - c.0 >= max(3, minHeight / 4) && seg[c.0..<c.1].max()! >= strong * ref
            }
            if cores.count <= 1 {
                lines.append((top, bottom))
                continue
            }
            var cuts = [top]
            for i in 0..<(cores.count - 1) {
                let e1 = cores[i].1, s2 = cores[i + 1].0
                var best = e1
                for y in e1..<s2 where seg[y] < seg[best] { best = y }
                cuts.append(top + best)
            }
            cuts.append(bottom)
            for i in 0..<(cuts.count - 1) where cuts[i + 1] - cuts[i] >= minHeight {
                lines.append((cuts[i], cuts[i + 1]))
            }
        }
        lines = splitTall(lines, smooth).filter { !isRule(raw, $0.0, $0.1) }

        var boxes: [LineBox] = []
        for (t0, b0) in lines {
            let t = max(0, t0 - pad), b = min(h, b0 + pad)
            var c0 = -1, c1 = -1
            for x in 0..<w {
                var any = false
                for y in t..<b where ink.bits[y * w + x] {
                    any = true
                    break
                }
                if any {
                    if c0 < 0 { c0 = x }
                    c1 = x
                }
            }
            if c0 < 0 { continue }
            boxes.append(LineBox(top: t, bottom: b, left: max(0, c0 - side), right: min(w, c1 + side)))
        }
        return boxes
    }

    // MARK: - model input

    static func lineInput(_ page: GrayImage, _ box: LineBox) -> (input: [Float], resized: GrayImage) {
        let h = box.bottom - box.top, w = box.right - box.left
        var crop = [UInt8](repeating: 0, count: w * h)
        for y in 0..<h {
            for x in 0..<w { crop[y * w + x] = page.pixels[(box.top + y) * page.width + box.left + x] }
        }
        return lineTensor(GrayImage(width: w, height: h, pixels: crop))
    }

    /// Model input for a cropped line image, exactly as train.py prepares it: mirrored, bicubic
    /// to height 64, scaled to [-1, 1], right edge repeated to width 800.
    public static func lineTensor(_ line: GrayImage) -> (input: [Float], resized: GrayImage) {
        let h = line.height, w = line.width
        var mirrored = [UInt8](repeating: 0, count: w * h)
        for y in 0..<h {
            for x in 0..<w { mirrored[y * w + x] = line.pixels[y * w + (w - 1 - x)] }
        }
        let rw = min(imgW, Int(ceil(Double(imgH) * (Double(w) / Double(h)))))
        let small = resizeBicubic(GrayImage(width: w, height: h, pixels: mirrored), rw, imgH)
        var x = [Float](repeating: 0, count: imgH * imgW)
        for y in 0..<imgH {
            for c in 0..<imgW {
                let v = Float(small.pixels[y * rw + min(c, rw - 1)])
                x[y * imgW + c] = (v / 255 - 0.5) / 0.5
            }
        }
        return (x, small)
    }

    // MARK: - Pillow's bicubic resize (Resample.c), bit for bit

    static func bicubic(_ value: Double) -> Double {
        let a = -0.5
        var x = value
        if x < 0.0 { x = -x }
        if x < 1.0 { return ((a + 2.0) * x - (a + 3.0)) * x * x + 1 }
        if x < 2.0 { return (((x - 5) * x + 8) * x - 4) * a }
        return 0.0
    }

    static func coeffs(_ inSize: Int, _ outSize: Int) -> (bounds: [(Int, Int)], kk: [Int], ksize: Int) {
        let scale = Double(inSize) / Double(outSize)
        let filterscale = max(scale, 1.0)
        let support = 2.0 * filterscale
        let ksize = Int(ceil(support)) * 2 + 1
        var bounds: [(Int, Int)] = []
        var kk = [Int](repeating: 0, count: outSize * ksize)
        for xx in 0..<outSize {
            let center = (Double(xx) + 0.5) * scale
            let ss = 1.0 / filterscale
            let xmin = max(0, Int(center - support + 0.5))          // Int() truncates like C
            let xmax = max(0, min(inSize, Int(center + support + 0.5)) - xmin)
            var k = [Double](repeating: 0, count: xmax)
            var ww = 0.0
            for x in 0..<xmax {
                let wgt = bicubic((Double(x + xmin) - center + 0.5) * ss)
                k[x] = wgt
                ww += wgt
            }
            for x in 0..<xmax {
                let v = ww != 0.0 ? k[x] / ww : k[x]
                kk[xx * ksize + x] = v < 0 ? Int(-0.5 + v * 4194304.0) : Int(0.5 + v * 4194304.0)
            }
            bounds.append((xmin, xmax))
        }
        return (bounds, kk, ksize)
    }

    @inline(__always) static func clip8(_ ss: Int) -> UInt8 {
        let v = ss >> 22
        return UInt8(v < 0 ? 0 : (v > 255 ? 255 : v))
    }

    /// Image.resize((outW, outH), Image.BICUBIC) for an 8-bit gray image.
    public static func resizeBicubic(_ img: GrayImage, _ outW: Int, _ outH: Int) -> GrayImage {
        var cur = img
        if outW != cur.width {
            let (bounds, kk, ksize) = coeffs(cur.width, outW)
            var px = [UInt8](repeating: 0, count: outW * cur.height)
            for y in 0..<cur.height {
                let row = y * cur.width
                for xx in 0..<outW {
                    let (xmin, xmax) = bounds[xx]
                    var ss = 1 << 21
                    for x in 0..<xmax { ss += Int(cur.pixels[row + xmin + x]) * kk[xx * ksize + x] }
                    px[y * outW + xx] = clip8(ss)
                }
            }
            cur = GrayImage(width: outW, height: cur.height, pixels: px)
        }
        if outH != cur.height {
            let (bounds, kk, ksize) = coeffs(cur.height, outH)
            var px = [UInt8](repeating: 0, count: cur.width * outH)
            for yy in 0..<outH {
                let (ymin, ymax) = bounds[yy]
                for x in 0..<cur.width {
                    var ss = 1 << 21
                    for y in 0..<ymax { ss += Int(cur.pixels[(ymin + y) * cur.width + x]) * kk[yy * ksize + y] }
                    px[yy * cur.width + x] = clip8(ss)
                }
            }
            cur = GrayImage(width: cur.width, height: outH, pixels: px)
        }
        return cur
    }
}
