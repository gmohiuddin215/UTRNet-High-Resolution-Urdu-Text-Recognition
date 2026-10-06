import CoreML
import CryptoKit
import Foundation

public enum UrduOCRError: Error {
    case noCharset
    case noOutput
}

public struct LineResult {
    public let box: LineBox
    public let text: String
}

public struct PageResult {
    /// Rotation applied to straighten the page, in degrees.
    public let skewDegrees: Double
    /// The deskewed page the line boxes refer to.
    public let page: GrayImage
    public let lines: [LineResult]
    /// One printed line per text line.
    public var text: String { lines.map(\.text).joined(separator: "\n") }
}

/// The exported UTRNet (UrduOCR.mlpackage from export_coreml.py) plus the page pipeline.
///
///     let ocr = try UrduOCRModel(contentsOf: Bundle.main.url(forResource: "UrduOCR", withExtension: "mlmodelc")!)
///     let result = try ocr.recognize(GrayImage(uiImage: scan))
///     print(result.text)
///
/// Recognition takes a few seconds per page: call it off the main thread.
public final class UrduOCRModel {
    public let model: MLModel
    /// Model class i is charset[i - 1]; class 0 is the CTC blank.
    public let charset: [String]
    /// Lines read per prediction, fixed at export (export_coreml.py --batch).
    public let batch: Int

    /// `url`: a compiled .mlmodelc (what Xcode puts in the app bundle) or an .mlpackage (compiled
    /// here first). `.cpuOnly` runs the float32 network exactly as verified by export_coreml.py;
    /// use `.cpuAndGPU` if the verification showed identical text there too.
    public init(contentsOf url: URL, computeUnits: MLComputeUnits = .cpuOnly) throws {
        var compiled = url
        if ["mlpackage", "mlmodel"].contains(url.pathExtension) {
            compiled = try MLModel.compileModel(at: url)
        }
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        model = try MLModel(contentsOf: compiled, configuration: config)
        let meta = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String]
        guard let json = meta?["charset"], let data = json.data(using: .utf8),
              let cs = try JSONSerialization.jsonObject(with: data) as? [String] else {
            throw UrduOCRError.noCharset
        }
        charset = cs
        batch = model.modelDescription.inputDescriptionsByName["image"]?
            .multiArrayConstraint?.shape.first?.intValue ?? 1
    }

    /// Read one prepared line (1 x 64 x 800 floats from Preprocess).
    public func read(_ input: [Float]) throws -> String {
        try read(lines: [input])[0]
    }

    /// Read prepared lines, `batch` per prediction (the last batch is padded).
    public func read(lines inputs: [[Float]]) throws -> [String] {
        let size = Preprocess.imgH * Preprocess.imgW
        var texts: [String] = []
        var start = 0
        while start < inputs.count {
            let n = min(batch, inputs.count - start)
            let shape = [batch, 1, Preprocess.imgH, Preprocess.imgW].map { NSNumber(value: $0) }
            let arr = try MLMultiArray(shape: shape, dataType: .float32)
            arr.withUnsafeMutableBufferPointer(ofType: Float.self) { buf, _ in   // new arrays are contiguous
                for b in 0..<batch {
                    let input = inputs[start + min(b, n - 1)]                    // pad with the last line
                    for i in 0..<size { buf[b * size + i] = input[i] }
                }
            }
            let features = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(multiArray: arr)])
            guard let logits = try model.prediction(from: features).featureValue(for: "logits")?.multiArrayValue else {
                throw UrduOCRError.noOutput
            }
            for b in 0..<n { texts.append(ctcDecode(argmax(logits, line: b))) }
            start += n
        }
        return texts
    }

    /// Best class at each of the 800 steps of line `b` (first maximum, like numpy argmax).
    func argmax(_ logits: MLMultiArray, line b: Int) -> [Int] {
        let steps = logits.shape[1].intValue, classes = logits.shape[2].intValue
        let st = logits.strides.map(\.intValue)
        var indices = [Int](repeating: 0, count: steps)
        if logits.dataType == .float32 {
            logits.withUnsafeBufferPointer(ofType: Float.self) { buf in
                for t in 0..<steps {
                    let row = b * st[0] + t * st[1]
                    var best = 0
                    var bestValue = buf[row]
                    for c in 1..<classes {
                        let v = buf[row + c * st[2]]
                        if v > bestValue {
                            bestValue = v
                            best = c
                        }
                    }
                    indices[t] = best
                }
            }
        } else {
            for t in 0..<steps {
                var best = 0
                var bestValue = logits[[NSNumber(value: b), NSNumber(value: t), 0]].floatValue
                for c in 1..<classes {
                    let v = logits[[NSNumber(value: b), NSNumber(value: t), NSNumber(value: c)]].floatValue
                    if v > bestValue {
                        bestValue = v
                        best = c
                    }
                }
                indices[t] = best
            }
        }
        return indices
    }

    /// Greedy CTC: drop repeats, then blanks.
    public func ctcDecode(_ indices: [Int]) -> String {
        var out = ""
        var prev = 0
        for i in indices {
            if i != 0 && i != prev { out += charset[i - 1] }
            prev = i
        }
        return out
    }

    /// Deskew, find lines, read them.
    public func recognize(_ gray: GrayImage, deskew: Bool = true) throws -> PageResult {
        let prepared = Preprocess.preparePage(gray, deskew: deskew)
        let texts = try read(lines: prepared.lines.map(\.input))
        let lines = zip(prepared.lines, texts).map { LineResult(box: $0.box, text: $1) }
        return PageResult(skewDegrees: prepared.skewDegrees, page: prepared.page, lines: lines)
    }
}

extension GrayImage {
    /// SHA-1 of the pixel bytes, as printed by `--dump` in ocr_page.py and urdu-ocr.
    public var sha1Hex: String {
        Insecure.SHA1.hash(data: Data(pixels)).map { String(format: "%02x", $0) }.joined()
    }
}
