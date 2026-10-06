// Hadith OCR screen for iPhone, iPad and Mac (Xcode "Multiplatform App" target).
//   Scan   - iPhone/iPad document camera (finds the page, removes perspective)
//   Photo  - a page image from Photos
//   File   - a page image or a PDF from Files / Finder
// Every page is deskewed, split into lines and read by UrduOCR.mlpackage (add it to the target;
// Xcode compiles it into UrduOCR.mlmodelc). iOS needs NSCameraUsageDescription in Info.

import PhotosUI
import SwiftUI
import UniformTypeIdentifiers
import UrduOCR
#if os(iOS)
import VisionKit
#endif

/// Loaded once, shared. CPU+GPU was verified to read exactly like PyTorch (export_coreml.py).
enum OCR {
    nonisolated(unsafe) static let model: Result<UrduOCRModel, Error> = Result {
        guard let url = Bundle.main.url(forResource: "UrduOCR", withExtension: "mlmodelc") else {
            throw CocoaError(.fileNoSuchFile)
        }
        return try UrduOCRModel(contentsOf: url, computeUnits: .cpuAndGPU)
    }
}

struct ContentView: View {
    @State private var text = ""
    @State private var busy = false
    @State private var status = ""
    @State private var importing = false
    @State private var photo: PhotosPickerItem?
    #if os(iOS)
    @State private var scanning = false
    #endif

    var body: some View {
        NavigationStack {
            ScrollView {
                Text(text.isEmpty ? "Scan a page, or choose a page image or PDF" : text)
                    .font(.custom("Jameel Noori Nastaleeq", size: 22, relativeTo: .body))
                    .multilineTextAlignment(.leading)                 // leading = right in RTL
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .environment(\.layoutDirection, .rightToLeft)
                    .textSelection(.enabled)
                    .padding()
            }
            .overlay { if busy { ProgressView(status).padding().background(.regularMaterial) } }
            .navigationTitle("Hadith OCR")
            .toolbar {
                #if os(iOS)
                ToolbarItem {
                    Button { scanning = true } label: { Label("Scan", systemImage: "doc.viewfinder") }
                        .disabled(busy || !VNDocumentCameraViewController.isSupported)
                }
                #endif
                ToolbarItem {
                    PhotosPicker(selection: $photo, matching: .images) {
                        Label("Photo", systemImage: "photo")
                    }
                    .disabled(busy)
                }
                ToolbarItem {
                    Button { importing = true } label: { Label("File", systemImage: "folder") }
                        .disabled(busy)
                }
                ToolbarItem {
                    ShareLink(item: text).disabled(text.isEmpty || busy)
                }
            }
            .fileImporter(isPresented: $importing, allowedContentTypes: [.pdf, .image]) { result in
                if case .success(let url) = result { openFile(url) }
            }
            .onChange(of: photo) { item in
                guard let item else { return }
                photo = nil
                Task {
                    guard let data = try? await item.loadTransferable(type: Data.self) else { return }
                    let url = FileManager.default.temporaryDirectory.appendingPathComponent("photo-\(UUID())")
                    try? data.write(to: url)
                    run(count: 1) { _ in try GrayImage(contentsOf: url) }
                }
            }
            #if os(iOS)
            .sheet(isPresented: $scanning) {
                DocumentScanner { pages in
                    scanning = false
                    if !pages.isEmpty { run(count: pages.count) { GrayImage(uiImage: pages[$0]) } }
                }
                .ignoresSafeArea()
            }
            #endif
        }
    }

    /// Copy the picked file into our sandbox (the permission to read it is temporary), then read it.
    func openFile(_ picked: URL) {
        let ok = picked.startAccessingSecurityScopedResource()
        defer { if ok { picked.stopAccessingSecurityScopedResource() } }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(UUID())-\(picked.lastPathComponent)")
        do {
            try FileManager.default.copyItem(at: picked, to: url)
        } catch {
            text = "Could not open \(picked.lastPathComponent): \(error.localizedDescription)"
            return
        }
        if url.pathExtension.lowercased() == "pdf" {
            run(count: GrayImage.pdfPageCount(url)) { try GrayImage(pdf: url, page: $0, dpi: 300) }
        } else {
            run(count: 1) { _ in try GrayImage(contentsOf: url) }
        }
    }

    /// OCR `count` pages off the main thread, showing the text as each page finishes.
    /// Long jobs keep the GPU busy for minutes and a phone has no fan, so when iOS reports the
    /// device as hot ("serious"), wait until it has cooled to "fair" before the next page.
    func run(count: Int, page: @escaping (Int) throws -> GrayImage) {
        guard count > 0 else { return }
        busy = true
        text = ""
        Task.detached(priority: .userInitiated) {
            do {
                let ocr = try OCR.model.get()
                var lastPage = ""
                for i in 0..<count {
                    while ProcessInfo.processInfo.thermalState.rawValue >= ProcessInfo.ThermalState.serious.rawValue {
                        await MainActor.run { status = "Phone is hot, cooling down before page \(i + 1)…" }
                        try await Task.sleep(nanoseconds: 10_000_000_000)
                    }
                    await MainActor.run {
                        status = (count > 1 ? "Reading page \(i + 1) of \(count)…" : "Reading…") + lastPage
                    }
                    let start = Date()
                    let result = try ocr.recognize(page(i))
                    lastPage = String(format: "\n(last page: %.1f s)", Date().timeIntervalSince(start))
                    let chunk = (count > 1 ? "--- page \(i + 1) ---\n" : "") + result.text
                    await MainActor.run { text += (text.isEmpty ? "" : "\n\n") + chunk }
                }
            } catch {
                await MainActor.run { text += "\n\nerror: \(error.localizedDescription)" }
            }
            await MainActor.run { busy = false }
        }
    }
}

#if os(iOS)
/// VisionKit's document camera: returns flat, cropped page images.
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
#endif
