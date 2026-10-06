// Command-line OCR on the Mac with the same Swift code the iPhone app uses.
//
//   swift run -c release urdu-ocr --model ../../UrduOCR.mlpackage page.png book.pdf
//   swift run -c release urdu-ocr --model ../../UrduOCR.mlpackage --dump page.png > swift.txt
//   python ocr_page.py page.png --model UrduOCR.mlpackage --dump > python.txt
//   diff python.txt swift.txt        # no output = the app prepares and reads the page identically

import CoreML
import Foundation
import UrduOCR

func usage() -> Never {
    print("""
    usage: urdu-ocr --model UrduOCR.mlpackage [--units cpu|gpu|all] [--dpi 300] [--no-deskew] [--dump] files...
      files: page images (PNG, JPEG, TIFF, HEIC) or PDFs
      --dump: print deskew angle, line boxes and input checksums (same format as ocr_page.py --dump)
    """)
    exit(2)
}

var modelPath: String?
var units = "cpu", dpi = 300.0, deskew = true, dump = false
var files: [String] = []
var args = Array(CommandLine.arguments.dropFirst())
func value() -> String {
    if args.isEmpty { usage() }
    return args.removeFirst()
}
while !args.isEmpty {
    let a = args.removeFirst()
    switch a {
    case "--model": modelPath = value()
    case "--units": units = value()
    case "--dpi":
        guard let d = Double(value()) else { usage() }
        dpi = d
    case "--no-deskew": deskew = false
    case "--dump": dump = true
    case "-h", "--help": usage()
    default: files.append(a)
    }
}
guard let modelPath, !files.isEmpty else { usage() }

let computeUnits: MLComputeUnits = units == "gpu" ? .cpuAndGPU : (units == "all" ? .all : .cpuOnly)

func process(_ label: String, _ gray: GrayImage, _ ocr: UrduOCRModel) throws {
    let t0 = Date()
    let prepared = Preprocess.preparePage(gray, deskew: deskew)
    let t1 = Date()
    let texts = try ocr.read(lines: prepared.lines.map(\.input))
    let t2 = Date()
    if dump {
        print("page \(label) skew \(prepared.skewStep) size \(prepared.page.width)x\(prepared.page.height) lines \(prepared.lines.count)")
        for (n, l) in prepared.lines.enumerated() {
            print("line \(n + 1) box \(l.box.top) \(l.box.bottom) \(l.box.left) \(l.box.right) width \(l.resized.width) sha1 \(l.resized.sha1Hex)")
        }
        for (n, t) in texts.enumerated() { print("text \(n + 1) \(t)") }
    } else {
        print("\n=== \(label)  (deskew \(String(format: "%+.2f", prepared.skewDegrees)) deg, \(prepared.lines.count) lines; "
              + String(format: "page prep %.1fs, reading %.1fs)", t1.timeIntervalSince(t0), t2.timeIntervalSince(t1)))
        print(texts.joined(separator: "\n"))
    }
}

do {
    let ocr = try UrduOCRModel(contentsOf: URL(fileURLWithPath: modelPath), computeUnits: computeUnits)
    _ = try ocr.read([Float](repeating: 0, count: Preprocess.imgH * Preprocess.imgW))   // load now, not in page 1's time
    for f in files {
        let url = URL(fileURLWithPath: f)
        if url.pathExtension.lowercased() == "pdf" {
            for p in 0..<GrayImage.pdfPageCount(url) {
                try process("\(url.lastPathComponent)#\(p + 1)", try GrayImage(pdf: url, page: p, dpi: dpi), ocr)
            }
        } else {
            try process(url.lastPathComponent, try GrayImage(contentsOf: url), ocr)
        }
    }
} catch {
    FileHandle.standardError.write("error: \(error)\n".data(using: .utf8)!)
    exit(1)
}
