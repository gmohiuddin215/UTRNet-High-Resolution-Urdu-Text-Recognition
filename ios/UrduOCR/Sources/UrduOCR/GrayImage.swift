import CoreGraphics
import Foundation
import ImageIO
#if canImport(UIKit)
import UIKit
#endif

/// 8-bit grayscale image, row-major, row 0 at the top.
public struct GrayImage {
    public let width: Int
    public let height: Int
    public var pixels: [UInt8]

    public init(width: Int, height: Int, pixels: [UInt8]) {
        precondition(pixels.count == width * height)
        self.width = width
        self.height = height
        self.pixels = pixels
    }

    @inline(__always) public subscript(x: Int, y: Int) -> UInt8 { pixels[y * width + x] }
}

public enum ImageLoadError: Error {
    case unreadable(URL)
    case noSuchPage(Int)
}

extension GrayImage {
    /// Same result as ocr_pipeline.load_gray: gray images are taken as they are, colour images
    /// go through Pillow's luma formula, transparent areas become white paper.
    public init(cgImage: CGImage) {
        let w = cgImage.width, h = cgImage.height
        let space = cgImage.colorSpace
        if space?.model == .monochrome {
            // draw into a gray context with the image's own colour space: no colour conversion
            var px = [UInt8](repeating: 255, count: w * h)
            px.withUnsafeMutableBytes { buf in
                let ctx = CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8,
                                    bytesPerRow: w, space: space!,
                                    bitmapInfo: CGImageAlphaInfo.none.rawValue)!
                ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
            }
            self.init(width: w, height: h, pixels: px)
            return
        }
        let rgbSpace = (space?.model == .rgb ? space : nil) ?? CGColorSpace(name: CGColorSpace.sRGB)!
        var rgba = [UInt8](repeating: 255, count: w * h * 4)          // starts as opaque white
        rgba.withUnsafeMutableBytes { buf in
            let ctx = CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8,
                                bytesPerRow: w * 4, space: rgbSpace,
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
        }
        var px = [UInt8](repeating: 0, count: w * h)
        for i in 0..<(w * h) {
            let r = UInt32(rgba[4 * i]), g = UInt32(rgba[4 * i + 1]), b = UInt32(rgba[4 * i + 2])
            px[i] = UInt8((r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16)
        }
        self.init(width: w, height: h, pixels: px)
    }

    /// A PNG/JPEG/TIFF/HEIC file (first frame). Use PNG when comparing with ocr_page.py: JPEG
    /// decoders differ slightly between libraries.
    public init(contentsOf url: URL) throws {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw ImageLoadError.unreadable(url)
        }
        self.init(cgImage: img)
    }

    /// Number of pages of a PDF file (0 if unreadable).
    public static func pdfPageCount(_ url: URL) -> Int {
        CGPDFDocument(url as CFURL)?.numberOfPages ?? 0
    }

    /// Render one PDF page (0-based) on white at `dpi`.
    public init(pdf url: URL, page index: Int, dpi: Double = 300) throws {
        guard let doc = CGPDFDocument(url as CFURL) else { throw ImageLoadError.unreadable(url) }
        guard let page = doc.page(at: index + 1) else { throw ImageLoadError.noSuchPage(index) }
        var box = page.getBoxRect(.mediaBox)
        if page.rotationAngle % 180 != 0 { box = CGRect(x: 0, y: 0, width: box.height, height: box.width) }
        let scale = dpi / 72.0
        let w = Int((box.width * scale).rounded()), h = Int((box.height * scale).rounded())
        var px = [UInt8](repeating: 255, count: w * h)
        px.withUnsafeMutableBytes { buf in
            let ctx = CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8,
                                bytesPerRow: w, space: CGColorSpaceCreateDeviceGray(),
                                bitmapInfo: CGImageAlphaInfo.none.rawValue)!
            ctx.setFillColor(gray: 1, alpha: 1)
            ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
            ctx.scaleBy(x: scale, y: scale)
            ctx.concatenate(page.getDrawingTransform(.mediaBox, rect: CGRect(origin: .zero, size: box.size),
                                                     rotate: 0, preserveAspectRatio: true))
            ctx.drawPDFPage(page)
        }
        self.init(width: w, height: h, pixels: px)
    }

    #if canImport(UIKit)
    /// A photo or a VisionKit document-camera scan, drawn upright first (camera images carry
    /// their rotation as metadata, not in the pixels).
    public init(uiImage: UIImage) {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        let size = CGSize(width: uiImage.size.width * uiImage.scale,      // full pixel resolution
                          height: uiImage.size.height * uiImage.scale)
        let upright = UIGraphicsImageRenderer(size: size, format: format).image { ctx in
            UIColor.white.setFill()
            ctx.fill(CGRect(origin: .zero, size: size))
            uiImage.draw(in: CGRect(origin: .zero, size: size))
        }
        self.init(cgImage: upright.cgImage!)
    }
    #endif
}
