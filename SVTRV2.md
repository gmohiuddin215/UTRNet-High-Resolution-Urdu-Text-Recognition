# Training SVTRv2 on the UTRNet data

[SVTRv2](https://arxiv.org/abs/2411.15858) is a transformer text recognizer with CTC output.
It is available here as `--FeatureExtraction SVTRv2`, so it trains on the same LMDB folders,
glyph list, augmentation and `--repeat_data` weighting as the fine-tuned UTRNet
([FINETUNE_ARABIC.md](FINETUNE_ARABIC.md)), and `eval_lines.py` scores both the same way.
This is step 0 of the NastaliqNet plan: does a plain SVTRv2 beat UTRNet on our books?

The model code is adapted from [OpenOCR](https://github.com/Topdu/OpenOCR) (Apache 2.0) in
`modules/svtrv2.py`; OpenOCR itself is not needed.

| File | Change |
|---|---|
| `modules/svtrv2.py` | SVTRv2 encoder and RCTC head (attention pooling over the line height) |
| `model.py` | `--FeatureExtraction SVTRv2`: encoder, RCTC head, linear CTC classifier |
| `train.py` | `--svtr_size`, `--svtr_upsample`, `--svtr_pretrained`; `--adamw`, `--weight_decay`, `--cosine`, `--warmup_epochs`, `--amp`; a check that the line gives enough CTC frames |
| `test.py`, `eval_lines.py`, `read.py` | accept `--svtr_size` and `--svtr_upsample` |

HRNet models train and behave exactly as before; the new optimizer flags are off by default.

## How it is set up for Urdu lines

* **Input 64x800, grey.** Same as the fine-tuned UTRNet. The first downsampling is chosen from
  `--imgH` (a multiple of 32), so a 64 or 96 px line reaches the global-attention stages at the
  same 8 rows as a 32 px line in the paper, keeping detail for dots and harakat until then.
* **CTC frames.** SVTRv2 keeps 1/4 of the width: 200 columns for an 800 px line, too few for a
  250-character vowelled Arabic label. `--svtr_upsample 2` (default) splits each column into 2
  frames, giving 400. `train.py` stops with a message if the frames are fewer than
  `--batch_max_length`.
* **RCTC head.** Each column attends over the line height to find its character instead of
  averaging the height away, so a mark above or below the baseline counts where it is.
* **No temporal-dropout ensemble.** UTRNet runs its BiLSTMs five times at inference; SVTRv2
  runs once.

| `--svtr_size` | parameters | CPU time per 64x800 line (inference) |
|---|---|---|
| tiny | 5.3M | 0.11 s |
| small | 11.6M | 0.12 s |
| base | 20.1M | 0.18 s |

Measured on this container's CPU; a GPU is far faster. GPU memory per batch has not been
measured yet: start with the batch sizes below and lower them on out-of-memory errors.

## Train

Use the LMDB folders from FINETUNE_ARABIC.md step 7 (synthetic, real and, if you have it,
UTRSet-Real). SVTRv2 starts from scratch, not from the UTRNet weights, so it needs more data
than fine-tuning did: use all of the synthetic lines.

```bash
COMMON="--FeatureExtraction SVTRv2 --svtr_size base --Prediction CTC \
        --charset UrduGlyphs_extended.txt --imgH 64 --imgW 800 --batch_max_length 250"

python train.py $COMMON --train_data lmdb/train --valid_data lmdb/val \
  --adamw --lr 5e-4 --weight_decay 0.05 --cosine --warmup_epochs 2 --amp \
  --batch_size 32 --repeat_data real:5 --num_epochs 30 --exp_name svtrv2_base
```

The learning rate, epochs and batch size are starting points, not tuned values. CTC models
usually output nothing for the first few epochs (validation accuracy stays at 0 while the loss
falls slowly) before they start reading; that is normal. If the loss is still flat after
about 5 epochs, lower `--lr` to `2e-4`. Do not use `--freeze_FE` (there is nothing pretrained
to keep) or the Adadelta default (transformers need AdamW).

On a 16 GB GPU try `--svtr_size small --batch_size 16`, or keep `base` with
`--batch_size 8 --accum_steps 4`.

### Optional: start from OpenOCR weights

OpenOCR publishes SVTRv2 checkpoints trained on English (Union14M) and Chinese text, linked in
its [SVTRv2 readme](https://github.com/Topdu/OpenOCR/tree/main/configs/rec/svtrv2). Download
the checkpoint of the same size (T, S or B for tiny, small, base) and pass it:

```bash
python train.py $COMMON ... --svtr_pretrained path/to/openocr_checkpoint.pth --exp_name svtrv2_base_pre
```

The log says how many tensors were loaded; if it is only a handful, the checkpoint is a
different SVTRv2 variant and gives no head start.

The encoder and head are loaded (the colour input layer is summed to one grey channel); the
classifier starts fresh. Latin or Chinese strokes help Urdu only a little, so train both ways
and keep the better one.

## Compare with UTRNet

Score both on the same held-out real lines, with the same flags the model was trained with:

```bash
python eval_lines.py --gt real_val/gt.txt --saved_model saved_models/svtrv2_base/best_norm_ED.pth \
  --FeatureExtraction SVTRv2 --svtr_size base --charset UrduGlyphs_extended.txt \
  --imgH 64 --imgW 800 --out svtrv2_val.tsv
python eval_lines.py --gt real_val/gt.txt --saved_model saved_models/stage2/best_norm_ED.pth \
  --charset UrduGlyphs_extended.txt --imgH 64 --imgW 800 --out utrnet_val.tsv
```

Compare the CER of each group (Urdu lines, Arabic lines, lines with harakat, honorifics), not
just the total. SVTRv2 replaces UTRNet only if it is better on the groups that matter for
your books.

## Not yet supported for SVTRv2

`ocr_page.py`, `deploy_model.py` and `export_coreml.py` are written for the HRNet model and
do not load SVTRv2 checkpoints yet. Use `eval_lines.py` and `read.py` until SVTRv2 has
proven better.
