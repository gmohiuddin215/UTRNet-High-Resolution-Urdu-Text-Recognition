# Fine-tuning UTRNet for Urdu hadith books: Arabic matn and honorifics

Tools for making UTRNet read Urdu books of hadith and other Islamic texts, e.g. Sunan Ibn Majah
with Urdu translation. Pages like these mix three things the released model handles badly:

* fully vowelled **Arabic matn** in a bold Naskh face,
* **honorifics** printed as a small sign or ligature (ﷺ, the ؓ over ابوہریرہ),
* Quranic quotations with pause marks and ornate brackets ﴿ ﴾.

The Urdu translation itself (Nastaliq) is what UTRNet was trained on and is already read well.

Licence reminder: UTRNet is CC BY-NC-SA 4.0, for research / personal use only.

| File | Purpose |
|---|---|
| `UrduGlyphs_extended.txt` | the original 180 glyphs + 29 the model cannot output yet, appended at the end |
| `expand_charset.py` | grows a checkpoint's output layer to the new glyph list, keeping every learned row |
| `check_font.py` | verifies a font can draw the charset and the honorific forms; renders previews |
| `make_lines.py` | synthetic lines: vowelled Naskh matn, honorific signs/ligatures, headers, footnotes |
| `page_to_lines.py` | cuts your corrected page scans into labeled line images |
| `eval_lines.py` | CER split into Urdu / Arabic / harakat / honorifics, plus the worst lines |
| `urdu_text.py` | shared label normalisation, honorific table, script detection |

Changes to the original scripts:

* `train.py`, `test.py`, `read.py`, `char_test.py` and `vis_salency.py` take `--charset`
  (default `UrduGlyphs.txt`), so you never overwrite the original list.
* `dataset.py` escapes the glyph list in its regex. Before this fix the filter for unknown
  characters silently matched nothing (the list contains `]` and `,-_`), and one unknown
  character in a label crashed training with a `KeyError`. It now reports how many labels were
  dropped for being too long or containing unknown characters.
* `train.py` also has `--repeat_data` (oversample a data folder), `--freeze_FE` (train only the
  BiLSTMs and output layer), `--freeze_bn` and `--accum_steps` (for small batches), a clear
  error when a checkpoint's class count does not match `--charset`, and `--batch_max_length`
  defaults to 250.
* Training runs on current PyTorch / NumPy (e.g. Colab). Before, it crashed in three places:
  an in-place addition in `modules/cnn/hrnet.py` broke `backward()` (fixed with bit-identical
  outputs, so the released weights behave exactly as before), `torch._utils._accumulate` no
  longer exists, and `imgaug` does not import under NumPy 2 (its three noise ops are replaced;
  `timm` is called with keyword arguments that work on old and new versions).
* Augmentation: the random rotation of up to 5 degrees pushed the ends of full-width lines out
  of the image (the label then no longer matched); it is now limited by the line's aspect ratio
  and fills with the background. Salt-and-pepper, border crop and resize are now drawn per image
  instead of once per dataset.

---

## Why the released model fails on these pages

1. **Missing characters.** `إ ى ٌ ٱ`, most Arabic-Indic digits (١٢٣٦٧٨٩), the Quranic pause marks
   `ۖ ۗ ۘ ۙ ۚ ۛ ۜ`, `۝ ۞ ۩` and `﴾ ﴿` are not in `UrduGlyphs.txt`, so the model cannot output them
   at all. `ى` and `إ` are everywhere in Arabic (على، إلى، عيسى).
2. **Input height of 32 px.** At that height harakat are 2-3 pixels tall and disappear. The
   model averages over the height (`AdaptiveAvgPool2d((None, 1))`), so the same weights accept a
   taller input: use `--imgH 64 --imgW 800`, which keeps the same width-to-height ratio as 32x400.
3. **Training data was Urdu Nastaliq.** The bold, fully vowelled Naskh of the matn and the
   small honorific clusters were never shown to it.
4. **Arabic labels are long.** A vowelled matn line is often 130-200 characters. `train.py`
   and `test.py` silently dropped labels longer than `--batch_max_length`, which defaulted to
   100: on test data that was 35% of all lines. The default is now 250 and the number of dropped
   labels is printed when the data loads.

### Label conventions

* **Honorifics are labelled as the full phrase** (صلی اللہ علیہ وسلم، رضی اللہ عنہ،
  رحمۃ اللہ علیہ، علیہ السلام), however the page prints them. This keeps output searchable and
  unambiguous. You may *type* the sign or ligature (ﷺ ؐ ؓ ؒ ؑ) in your transcriptions:
  `page_to_lines.py` and `eval_lines.py` turn them into the phrase (see `urdu_text.py`).
* Type exactly what is printed: if the book writes اَبِیْ with Urdu ی, do not "correct" it to
  أبي. The model can only learn what it sees. For Indo-Pak prints of Arabic (as in the
  Jahangiri Sunan Ibn Majah) that means:
  * alef with hamza is printed as a bare alef carrying the vowel: اَبُوْ، اِذَا، اَنْبَاَنَا
  * a final ya without dots is ی (U+06CC), also for alef maqsura: اَبِیْ، عَلٰی، صَلَّی؛ a ya
    with dots is ي (U+064A): شَيْبَةَ، يَقُوْلُ
  * long vowels carry sukun when printed: رَسُوْلُ، فِیْ
  * the name of Allah with shadda and standing alef: اللّٰهِ
  * Arabic letters in Arabic text (ه ك ة), Urdu letters in Urdu text (ہ ک ی ے)
  * idgham shadda where printed: طَائِفَةٌ مِّنْ، خَطًّا وَّخَطَّ
  * a line that cannot be labelled cleanly (a rule, a smudge, a footnote wrapped onto two rows
    in one band, a damaged line) gets `#`
* Text is NFC-normalised and zero-width characters (ZWNJ, direction marks) and tatweel are
  removed, so the same printed word always gets the same label.

---

## 0. Setup

```bash
python3 -m venv utrnet && source utrnet/bin/activate
pip install torch torchvision timm pillow numpy opencv-python lmdb natsort nltk fire fonttools pytz six matplotlib tqdm
python -c "from PIL import features; print('raqm:', features.check('raqm'))"   # must be True
```
If RAQM is `False`, Arabic and Nastaliq render as unjoined letters. On macOS:
`brew install libraqm` and reinstall Pillow; `check_font.py` tells you if it is missing.

## 1. The free test (no training)

Crop a few Arabic lines from a page (or run step 5 on one page) and compare the released model
at the two heights. The released model uses the **original** glyph list:

```bash
python eval_lines.py --gt real_lines/gt.txt --saved_model UTRNet-Large.pth \
    --charset UrduGlyphs.txt --imgH 32 --imgW 400
python eval_lines.py --gt real_lines/gt.txt --saved_model UTRNet-Large.pth \
    --charset UrduGlyphs.txt --imgH 64 --imgW 800
```
If "Arabic lines" and "lines with harakat" improve at 64, the height diagnosis is confirmed.
(Errors on `ى إ ٌ` remain in both runs: that model cannot output them.)

For a single image: `python read.py --image_path line.png --saved_model UTRNet-Large.pth --imgH 64 --imgW 800`.

## 2. Fonts

You need a Nastaliq face (Jameel Noori Nastaleeq, Noto Nastaliq Urdu) and one or two bold Naskh
faces close to your books' matn (Noto Naskh Arabic Bold, Scheherazade New Bold, an Indo-Pak
Quran font if you scan Quranic text). Check each one:

```bash
python check_font.py --font fonts/JameelNooriNastaleeq.ttf --out font_preview_ur
python check_font.py --font fonts/NotoNaskhArabic-Bold.ttf --out font_preview_ar
```
It lists how much of `UrduGlyphs_extended.txt` the font covers and which honorific forms it can
print (ligature glyph, sign over the name, or only small text). Open the PNGs: `arabic_matn.png`
must show clear harakat, and `signs.png` / `ligatures.png` should look like your books.
`make_lines.py` never draws a character the font lacks (it tries the other fonts in the pool,
then skips the line), so a gap is not fatal, just less variety.

## 3. Corpus

* `arabic_matn.txt`: hadith isnad and matn, one sentence per line, ideally from the collections you
  are scanning, vowelled as printed. Lines are rendered exactly as written: make_lines.py never
  adds harakat, never appends text to a hadith or verse, and never joins two passages, so use a
  verified source (wrong text in the corpus means wrong text in the images).
* `urdu.txt`: Urdu translation text from similar books.

A few MB of each is plenty. Matching the domain matters more than the amount.

## 4. Expand the checkpoint

```bash
python expand_charset.py --ckpt UTRNet-Large.pth --out UTRNet-Large-extended.pth
```
Expected output: `copied 182/211 rows (blank + 181 known glyphs), 29 new rows initialised`.

## 5. Real lines from your own pages

Synthetic print never quite matches a scanned book, so correct some real pages. Scan at 300 DPI.
For each `page.png` write `page.txt` with **one printed line per text line**, in reading order,
header and footnotes included (`#` alone on a line skips that line).

```bash
python page_to_lines.py --data my_pages --preview_only   # check the numbered boxes
python page_to_lines.py --data my_pages --out real_lines --charset UrduGlyphs_extended.txt
```
On the sample Ibn Majah page (2480x3509) it finds all 24 printed lines, including large
titles and the short last line of a hadith, erases the header and footnote rules, and deskews by
+0.75°. Open `real_lines/page_previews/*.png`: every printed line should have exactly one box.
If two lines share a box, lower `--split`; if one line is cut in two, raise `--strong` or
`--min_gap`. Pages where the box count and text-line count differ are skipped and listed, never
silently misaligned. Crops keep the scan resolution; training resizes them.

**Keep some pages apart for validation.** Put their lines in a separate folder, so validation
measures new pages rather than lines the model has seen.

## 6. Synthetic lines

```bash
python make_lines.py \
  --urdu_fonts fonts/JameelNooriNastaleeq.ttf fonts/NotoNastaliqUrdu.ttf \
  --arabic_fonts fonts/NotoNaskhArabic-Bold.ttf fonts/ScheherazadeNew-Bold.ttf \
  --urdu_corpus urdu.txt --arabic_corpus arabic_matn.txt \
  --n 120000 --imgH 64 --out synth_train
python make_lines.py ...same fonts and corpus... --n 4000 --seed 99 --out synth_val
```
Defaults: 35% Arabic matn, 25% Urdu with honorifics, 10% mixed, plus headers, footnotes,
references and plain Urdu. Honorifics are drawn as the font's ligature (ﷺ), as the sign over
the preceding name (ابوہریرہؓ), or as small raised text, and always labelled with the full phrase.
Lines with harakat get gentler degradation and a larger minimum font size, so the marks the
label promises stay visible. **Look at 20 images before generating 120k.**

## 7. Build the LMDB datasets

```bash
python create_lmdb_dataset.py --inputPath synth_train --gtFile synth_train/gt.txt --outputPath lmdb/train/synth
python create_lmdb_dataset.py --inputPath real_lines  --gtFile real_lines/gt.txt  --outputPath lmdb/train/real
python create_lmdb_dataset.py --inputPath synth_val   --gtFile synth_val/gt.txt   --outputPath lmdb/val/synth
python create_lmdb_dataset.py --inputPath real_val    --gtFile real_val/gt.txt    --outputPath lmdb/val/real
```
`train.py` uses every LMDB folder under `--train_data`. If you have UTRSet-Real, add it as
`lmdb/train/utrset` so the model keeps its Urdu while it learns Arabic.

## 8. Train (GPU: Colab T4 or a rented card, not a laptop)

**Memory decides the batch size.** HRNet keeps full resolution, so a 64x800 line costs far more
than the original 32x400. Measured training memory per image (float32):

| setting | per image | 16 GB GPU (T4) | 24 GB (L4, A10) | 40 GB (A100) |
|---|---|---|---|---|
| 64x800, whole network | ~3.7 GB | batch 3 | batch 5 | batch 9 |
| 64x800, `--freeze_FE` | ~0.9 GB | batch 12 | batch 20 | batch 32 |
| 32x400, whole network (original) | ~1.0 GB | batch 12 | batch 20 | batch 32 |

With small batches, add `--freeze_bn` (keeps the pretrained BatchNorm statistics, which a batch
of 3 would only make noisy) and `--accum_steps` (sums gradients over several batches, so
batch 3 x 4 steps behaves like a batch of 12 for the optimizer).

```bash
COMMON="--FeatureExtraction HRNet --SequenceModeling DBiLSTM --Prediction CTC \
        --charset UrduGlyphs_extended.txt --imgH 64 --imgW 800 --batch_max_length 250"

# stage 1 (warm-up): CNN frozen, the BiLSTMs and the 29 new output rows learn first
python train.py $COMMON --train_data lmdb/train --valid_data lmdb/val \
  --saved_model UTRNet-Large-extended.pth --FT --freeze_FE --adam --lr 3e-4 \
  --repeat_data real:3 --batch_size 12 --num_epochs 3 --exp_name stage1

# stage 2: whole network, small batches, real lines weighted up
python train.py $COMMON --train_data lmdb/train --valid_data lmdb/val \
  --saved_model saved_models/stage1/best_norm_ED.pth --FT --adam --lr 5e-5 \
  --freeze_bn --batch_size 3 --accum_steps 4 \
  --repeat_data real:10 --num_epochs 10 --exp_name stage2
```
Batch sizes are for a 16 GB GPU; scale them with the table. Learning rates and epochs are
starting points, not tuned values: watch the validation numbers in
`saved_models/<exp>/log_train.txt` and stop when they stop improving. `--repeat_data real:10`
counts every real line ten times per epoch (training set only). The best model is saved as
`saved_models/<exp>/best_norm_ED.pth`. Out of memory: lower `--batch_size` and raise
`--accum_steps` by the same factor.

## 9. Measure where it still fails

```bash
python eval_lines.py --gt real_val/gt.txt --saved_model saved_models/stage2/best_norm_ED.pth \
  --charset UrduGlyphs_extended.txt --imgH 64 --imgW 800 --out stage2_val.tsv
```
You get CER for all lines, Urdu lines, Arabic lines, lines with harakat and lines with
honorifics; the same CER with harakat ignored; how many honorific phrases came out exactly; and
the worst lines with reference and prediction. Work on the weakest group, not the average:

* Arabic CER high, "no harakat" CER also high → more matn lines, more Naskh fonts close to your books
* Arabic CER high but "no harakat" CER low → the errors are marks: try `--imgH 96 --imgW 1200`
  for both training and inference, and check harakat are visible in the synthetic images
* honorifics wrong → raise `--p_urdu_honorific`; check `check_font.py` shows the forms your books use
* real lines much worse than synthetic → correct more pages; real lines matter most at the end

## 10. Inference must match training

Always pass the same `--charset`, `--imgH` and `--imgW` you trained with. A mismatch either
fails loudly (wrong charset) or silently costs accuracy (wrong size).

```bash
python read.py --image_path line.png --saved_model saved_models/stage2/best_norm_ED.pth \
  --charset UrduGlyphs_extended.txt --imgH 64 --imgW 800
```

---

## Expectations

Urdu is already read well, so the work is Arabic and honorifics. Fixing the charset and the
input height and training on Naskh-heavy synthetic data should give a large jump on Arabic
lines. Real corrected lines take it the rest of the way. Getting to one or two slips per page
(about 1% CER on Arabic lines) normally needs both stages and a few thousand real lines: plan
for two or three rounds of correcting pages, retraining and measuring.

For whole pages you still need line detection: `page_to_lines.py` handles clean book scans,
not arbitrary layouts. The authors' End-To-End-Urdu-OCR-WebApp pairs UTRNet with a YOLOv8 detector.
