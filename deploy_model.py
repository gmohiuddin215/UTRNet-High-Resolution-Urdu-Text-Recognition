"""
The trained UTRNet as a deterministic function, for export and for ocr_page.py.

At test time model.py keeps its temporal dropout on: it draws 5 random masks over the 800 time
steps on every call and averages the 5 LSTM outputs, so the same line can read differently from
run to run, and Core ML cannot repeat numpy's random draws. DeployModel freezes the 5 masks
(the ones eval_lines.py draws for its first batch: numpy seed 0) and stores them in the model,
so PyTorch, Core ML on the Mac and Core ML on the phone all compute the same thing.
"""
import types

import numpy as np
import torch
import torch.nn as nn

from model import Model

IMG_H, IMG_W, N_MASKS = 64, 800, 5


def load_charset(path):
    from urdu_text import load_charset as _load
    return _load(path)


def build_opt(charset):
    return types.SimpleNamespace(
        FeatureExtraction="HRNet", SequenceModeling="DBiLSTM", Prediction="CTC",
        input_channel=1, output_channel=32, hidden_size=256, num_fiducial=20,
        batch_max_length=250, imgH=IMG_H, imgW=IMG_W, rgb=False, device="cpu",
        character=charset, num_class=len(charset) + 1)


def load_trained(checkpoint, charset):
    """model.py's Model with the checkpoint's weights, in eval mode, on the CPU."""
    opt = build_opt(charset)
    model = Model(opt)
    state = torch.load(checkpoint, map_location="cpu")
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    w = state.get("Prediction.weight")
    if w is not None and w.shape[0] != opt.num_class:
        raise SystemExit(f"{checkpoint} predicts {w.shape[0]} classes but the charset gives "
                         f"{opt.num_class}: use the charset the checkpoint was trained with.")
    model.load_state_dict(state)
    return model.eval()


def frozen_masks(seed=0):
    rng = np.random.RandomState(seed)
    return torch.tensor(np.stack([(rng.rand(IMG_W) > 0.2) for _ in range(N_MASKS)]),
                        dtype=torch.float32)                             # [5, 800]


class FixedUpsample(nn.Module):
    """nn.Upsample(scale_factor=s, bilinear) for one known input size. Same output (integer
    factors map coordinates identically by size or by factor), but the size is a constant, so the
    Core ML converter does not have to turn traced shape arithmetic into integers."""

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, mode="bilinear", align_corners=False)


def freeze_upsampling(feature_extractor):
    sizes, hooks = {}, []
    ups = [(name, m) for name, m in feature_extractor.named_modules() if isinstance(m, nn.Upsample)]
    for name, m in ups:
        hooks.append(m.register_forward_hook(
            lambda mod, inp, out, name=name: sizes.__setitem__(name, tuple(out.shape[2:]))))
    with torch.no_grad():
        feature_extractor(torch.zeros(1, 1, IMG_H, IMG_W))
    for h in hooks:
        h.remove()
    for name, _ in ups:
        parent_name, _, child = name.rpartition(".")
        parent = feature_extractor.get_submodule(parent_name) if parent_name else feature_extractor
        setattr(parent, child, FixedUpsample(sizes[name]))


class DeployModel(nn.Module):
    """image [B, 1, 64, 800] in [-1, 1], mirrored -> logits [B, 800, classes] (class 0 = blank)."""

    def __init__(self, model, masks=None):
        super().__init__()
        freeze_upsampling(model.FeatureExtraction)
        self.FeatureExtraction = model.FeatureExtraction
        self.SequenceModeling = model.SequenceModeling
        self.Prediction = model.Prediction
        self.register_buffer("masks", frozen_masks() if masks is None else masks)

    def forward(self, image):
        f = self.FeatureExtraction(image)                                # [B, 32, 64, 800]
        v = f.mean(dim=2).permute(0, 2, 1)                               # [B, 800, 32]
        m = self.masks[:, None, :, None]                                 # [5, 1, 800, 1]
        x = (v.unsqueeze(0) * m).flatten(0, 1)                           # 5 masked copies
        for block in self.SequenceModeling:      # BidirectionalLSTM.forward minus flatten_parameters(),
            x = block.linear(block.rnn(x)[0])    # which a CUDA-only no-op the exporters cannot trace
        y = x.reshape(N_MASKS, -1, IMG_W, self.Prediction.in_features)   # no traced shapes:
                                                                         # Core ML wants constants
        y = (y[0] + (y[1] + (y[2] + (y[3] + y[4])))) * (1 / 5)          # same order as model.py
        return self.Prediction(y)


def load_deploy(checkpoint, charset):
    return DeployModel(load_trained(checkpoint, charset)).eval()
