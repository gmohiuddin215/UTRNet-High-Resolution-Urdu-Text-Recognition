# Running the hadith OCR model on Mac, iPhone and iPad

Three pieces, all reading pages the same way:

| piece | what it does |
|---|---|
| `export_coreml.py` | turns `utrnet_hadith_final.pth` into `UrduOCR.mlpackage` (full float32) and checks on your labelled lines that Core ML reads exactly what PyTorch reads |
| `ocr_page.py` | OCRs page images or PDFs on the Mac: deskew → find lines → read → `.txt` |
| `ios/UrduOCR` | Swift package with the same pipeline for the iPhone/iPad/Mac app, plus a `urdu-ocr` command-line tool |

Every page is **deskewed first**: the angle (up to ±15°) whose row profile of the text is sharpest is
found in 0.05° steps, ignoring rules, frames and page edges, and the page is rotated by it before
lines are cut. On the phone the document camera also flattens perspective before that.

## 1. Mac setup (once)

```bash
git clone -b claude/project-understanding-bef1fc https://github.com/gmohiuddin215/UTRNet-High-Resolution-Urdu-Text-Recognition.git
cd UTRNet-High-Resolution-Urdu-Text-Recognition
python3 -m venv ~/utrnet-env
source ~/utrnet-env/bin/activate          # run this again in every new terminal
pip install torch==2.7.0 coremltools==9.0 numpy pillow pymupdf
cp -R ~/Downloads/results .               # utrnet_hadith_final.pth + UrduGlyphs_extended.txt
```

coremltools 9.0 is tested with torch 2.7; newer torch versions fail to convert this model.

## 2. Export to Core ML and verify

```bash
python export_coreml.py --saved_model results/utrnet_hadith_final.pth \
    --charset results/UrduGlyphs_extended.txt --out UrduOCR.mlpackage \
    --gt khatme/val/gt.txt --page some_page.png
```

`--gt` takes any `image<TAB>label` file (e.g. `khatme/val/gt.txt` from the khatme zips); `--page`
runs whole pages through deskew and line finding too. Both can be repeated. On a slow Mac add
`--limit 50`.

The table at the end is the proof:

```
model                                  identical text  max |logit diff|  argmax match   CER  lenient
PyTorch, model.py (random masks)         ...
PyTorch, deterministic (reference)       147/147              -              -        ...
Core ML UrduOCR.mlpackage, CPU           147/147         …             100.0000%     ...
Core ML UrduOCR.mlpackage, CPU+GPU       147/147         ...
```

* **identical text N/N on CPU** means the app reads every line exactly as PyTorch does. The app
  uses the CPU by default for this reason.
* If CPU+GPU is also N/N, the app can use `computeUnits: .cpuAndGPU`, which is faster.
* The first row is `model.py` as it ran in training. That model changes its temporal-dropout masks on
  every call, so the same line can read slightly differently from run to run. The exported model
  freezes the masks, so it is repeatable. Its CER should match the training numbers within noise.
* `--fp16` also writes `UrduOCR_fp16.mlpackage`: half the size and able to use the Neural Engine, but
  not lossless. The same table shows how many lines it changes.

## 3. OCR pages on the Mac

```bash
python ocr_page.py book.pdf --model UrduOCR.mlpackage                  # all pages
python ocr_page.py book.pdf --model UrduOCR.mlpackage --pages 12-30 --preview
python ocr_page.py scans/ --model results/utrnet_hadith_final.pth --charset results/UrduGlyphs_extended.txt
```

The text goes to `ocr_output/<name>.txt`, one printed line per text line, with `--- page N ---`
between pages. `--preview` saves each deskewed page with numbered line boxes. Check it when a page
reads badly: usually two lines got merged or a line was split. The `.mlpackage` and the `.pth` give
the same text.

## 4. iPhone / iPad app

1. In Xcode, create a new **App** (SwiftUI), iOS 16 or later.
2. **File → Add Package Dependencies… → Add Local…** and choose `ios/UrduOCR`. Add the `UrduOCR`
   library to the app target.
3. Drag `UrduOCR.mlpackage` into the project (target membership: the app). Xcode compiles it into
   `UrduOCR.mlmodelc` inside the app.
4. Replace `ContentView.swift` with `ios/Example/ContentView.swift`.
5. Info → add **Privacy - Camera Usage Description** (`NSCameraUsageDescription`).
6. Optional, to display the result in Nastaliq: add `JameelNooriNastaleeq-Regular.ttf` to the
   target and list it under **Fonts provided by application** (`UIAppFonts`).
7. Run it on a device (the document camera does not work in the Simulator).

Using the library directly:

```swift
import UrduOCR
let ocr = try UrduOCRModel(contentsOf: Bundle.main.url(forResource: "UrduOCR", withExtension: "mlmodelc")!)
let page = try ocr.recognize(GrayImage(uiImage: image))     // or GrayImage(contentsOf:), GrayImage(pdf:page:)
print(page.skewDegrees, page.text)
for line in page.lines { print(line.box, line.text) }
```

The same package works in a macOS app. `GrayImage(cgImage:)`, `GrayImage(contentsOf:)` and
`GrayImage(pdf:page:dpi:)` are available on both platforms.

## 5. Check that the app prepares pages like the Python script

```bash
cd ios/UrduOCR
swift run -c release urdu-ocr --model ../../UrduOCR.mlpackage --dump ../../page.png > /tmp/swift.txt
cd ../..
python ocr_page.py page.png --model UrduOCR.mlpackage --dump > /tmp/python.txt
diff /tmp/python.txt /tmp/swift.txt && echo identical
```

`--dump` prints the deskew angle, every line box, and a checksum of every model input. No diff
output means the Swift pipeline built exactly the same inputs and read the same text. Use a
**PNG** for this check. JPEG decoders and PDF renderers differ slightly between Apple's frameworks
and the Python libraries, so a JPEG or PDF can differ by a pixel here and there. That is like
scanning the page twice, not a loss in the model.

## What "no loss" covers, and what it doesn't

* **Model:** the Core ML package holds the same float32 weights and computes the same network. Step 2
  measures this on your own lines.
* **Pipeline:** the Swift preprocessing is a line-by-line port of `ocr_pipeline.py`. The bicubic
  resize reproduces Pillow bit for bit, which is what training used. Step 5 measures this.
* **Speed:** float32 cannot run on the Neural Engine. Expect roughly 0.2–1 s per line on recent
  iPhones and Macs, so a 25-line page takes a few seconds. `--fp16` is the faster option if its row in
  the table is acceptable to you.
* **Page layout:** lines are found by horizontal projection, the same way the training data was cut.
  Single-column pages work. Two-column pages, or photos with heavy curl or a dark background, need
  cropping first. On the phone the document camera does that cropping.
