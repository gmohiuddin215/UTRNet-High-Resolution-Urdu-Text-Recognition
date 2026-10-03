// Minimal iPhone/iPad screen: scan a page with the document camera, OCR it, show the text.
// Add to an Xcode app target that depends on the UrduOCR package and contains UrduOCR.mlpackage
// (Xcode compiles it into UrduOCR.mlmodelc in the app bundle). Info.plist needs
// NSCameraUsageDescription.

import SwiftUI
import UrduOCR
import VisionKit

struct ContentView: View {
    @State private var scanning = false
    @State private var busy = false
    @State private var text = ""

    var body: some View {
        NavigationStack {
            ScrollView {
                Text(text.isEmpty ? "Scan a page to read it" : text)
                    .font(.custom("Jameel Noori Nastaleeq", size: 22, relativeTo: .body))
                    .multilineTextAlignment(.trailing)
                    .environment(\.layoutDirection, .rightToLeft)
                    .frame(maxWidth: .infinity, alignment: .trailing)
                    .textSelection(.enabled)
                    .padding()
            }
            .overlay { if busy { ProgressView("Reading…") } }
            .toolbar {
                Button("Scan") { scanning = true }.disabled(busy)
                ShareLink(item: text).disabled(text.isEmpty)
            }
            .sheet(isPresented: $scanning) {
                DocumentScanner { pages in
                    scanning = false
                    read(pages)
                }
                .ignoresSafeArea()
            }
        }
    }

    func read(_ pages: [UIImage]) {
        busy = true
        Task.detached(priority: .userInitiated) {
            var out: [String] = []
            do {
                let url = Bundle.main.url(forResource: "UrduOCR", withExtension: "mlmodelc")!
                let ocr = try UrduOCRModel(contentsOf: url)                // .cpuOnly: exact float32
                for page in pages {
                    out.append(try ocr.recognize(GrayImage(uiImage: page)).text)
                }
            } catch {
                out.append("error: \(error)")
            }
            let result = out.joined(separator: "\n\n")
            await MainActor.run {
                text = result
                busy = false
            }
        }
    }
}

/// VisionKit's document camera: finds the page edges, removes perspective and returns flat
/// page images; UrduOCR then straightens any remaining tilt.
struct DocumentScanner: UIViewControllerRepresentable {
    let done: ([UIImage]) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(done: done) }

    func makeUIViewController(context: Context) -> VNDocumentCameraViewController {
        let vc = VNDocumentCameraViewController()
        vc.delegate = context.coordinator
        return vc
    }

    func updateUIViewController(_ vc: VNDocumentCameraViewController, context: Context) {}

    final class Coordinator: NSObject, VNDocumentCameraViewControllerDelegate {
        let done: ([UIImage]) -> Void
        init(done: @escaping ([UIImage]) -> Void) { self.done = done }

        func documentCameraViewController(_ controller: VNDocumentCameraViewController,
                                          didFinishWith scan: VNDocumentCameraScan) {
            done((0..<scan.pageCount).map { scan.imageOfPage(at: $0) })
        }

        func documentCameraViewControllerDidCancel(_ controller: VNDocumentCameraViewController) {
            done([])
        }

        func documentCameraViewController(_ controller: VNDocumentCameraViewController,
                                          didFailWithError error: Error) {
            done([])
        }
    }
}
