// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "UrduOCR",
    platforms: [.iOS(.v16), .macOS(.v13)],
    products: [
        .library(name: "UrduOCR", targets: ["UrduOCR"]),
        .executable(name: "urdu-ocr", targets: ["urdu-ocr"]),
    ],
    targets: [
        .target(name: "UrduOCR"),
        .executableTarget(name: "urdu-ocr", dependencies: ["UrduOCR"]),
    ]
)
