"""
SVTRv2 encoder and RCTC feature head, for use as --FeatureExtraction SVTRv2.

Adapted from OpenOCR (https://github.com/Topdu/OpenOCR, commit 1ccfc6e):
openrec/modeling/encoders/svtrv2_lnconv_two33.py, openrec/modeling/decoders/rctc_decoder.py
and openrec/modeling/common.py. Copyright (c) Topdu, Apache License 2.0
(http://www.apache.org/licenses/LICENSE-2.0). Paper: "SVTRv2: CTC Beats Encoder-Decoder Models
in Scene Text Recognition", https://arxiv.org/abs/2411.15858

Changes from OpenOCR:
* attention uses F.scaled_dot_product_attention (same result, far less memory on 64 px lines)
* the first downsampling is chosen from the input height, so 64 or 96 px lines reach the
  global-attention stages at the same 8 rows as a 32 px line does in the paper
* RCTCHead is RCTCDecoder without its classifier (the classifier is model.Prediction, as in
  UTRNet) and splits each output column into `upsample` CTC frames: SVTRv2 keeps 1/4 of the
  width, and an 800 px line then has 200 columns, fewer than a 250-character vowelled Arabic
  label needs
* module names follow OpenOCR, so OpenOCR SVTRv2 weights can be loaded with load_openocr()
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, ones_, trunc_normal_, zeros_

# dims, depths, heads and mixers of OpenOCR's svtrv2_{tiny,small,}_rctc.yml
PRESETS = {
    'tiny': dict(dims=[64, 128, 256], depths=[3, 6, 3], num_heads=[2, 4, 8],
                 mixer=[['Conv'] * 3, ['Conv'] * 3 + ['FGlobal', 'Global', 'Global'], ['Global'] * 3]),
    'small': dict(dims=[96, 192, 384], depths=[3, 6, 3], num_heads=[3, 6, 12],
                  mixer=[['Conv'] * 3, ['Conv'] * 3 + ['FGlobal', 'Global', 'Global'], ['Global'] * 3]),
    'base': dict(dims=[128, 256, 384], depths=[6, 6, 6], num_heads=[4, 8, 12],
                 mixer=[['Conv'] * 6, ['Conv', 'Conv', 'FGlobal', 'Global', 'Global', 'Global'], ['Global'] * 6]),
}


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    return x * x.new_empty(shape).bernoulli_(keep_prob).div_(keep_prob)


class DropPath(nn.Module):

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop if self.training else 0.0)
        return self.proj_drop(self.proj(x.transpose(1, 2).reshape(B, N, C)))


class Block(nn.Module):
    """Post-norm transformer block of the SVTRv2 encoder."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, eps=1e-6):
        super().__init__()
        self.norm1 = norm_layer(dim, eps=eps)
        self.mixer = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim, eps=eps)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer, drop)

    def forward(self, x):
        x = self.norm1(x + self.drop_path(self.mixer(x)))
        return self.norm2(x + self.drop_path(self.mlp(x)))


class PreNormBlock(nn.Module):
    """Pre-norm transformer block (OpenOCR common.Block), used by the RCTC head."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ConvBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, drop_path=0.0, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, eps=1e-6, num_conv=2, kernel_size=3):
        super().__init__()
        self.norm1 = norm_layer(dim, eps=eps)
        self.mixer = nn.Sequential(*[nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2, groups=num_heads)
                                     for _ in range(num_conv)])
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim, eps=eps)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer, drop)

    def forward(self, x):
        C, H, W = x.shape[1:]
        x = x + self.drop_path(self.mixer(x))
        x = self.norm1(x.flatten(2).transpose(1, 2))
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x.transpose(1, 2).reshape(-1, C, H, W)


class FlattenTranspose(nn.Module):

    def forward(self, x):
        return x.flatten(2).transpose(1, 2)


class SubSample2D(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, sz):
        x = self.conv(x)
        C, H, W = x.shape[1:]
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x.transpose(1, 2).reshape(-1, C, H, W), [H, W]


class SubSample1D(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, sz):
        x = x.transpose(1, 2).reshape(-1, x.shape[-1], sz[0], sz[1])
        x = self.conv(x)
        C, H, W = x.shape[1:]
        return self.norm(x.flatten(2).transpose(1, 2)), [H, W]


class IdentitySize(nn.Module):

    def forward(self, x, sz):
        return x, sz


class SVTRStage(nn.Module):

    def __init__(self, dim, out_dim, depth, mixer, sub_k, num_heads, mlp_ratio, qkv_bias,
                 drop_path, num_conv, downsample):
        super().__init__()
        self.blocks = nn.Sequential()
        for i in range(depth):
            if mixer[i] == 'Conv':
                self.blocks.append(ConvBlock(dim, num_heads, mlp_ratio, drop_path=drop_path[i], num_conv=num_conv))
            else:
                if mixer[i] == 'FGlobal':
                    self.blocks.append(FlattenTranspose())
                self.blocks.append(Block(dim, num_heads, mlp_ratio, qkv_bias, drop_path=drop_path[i]))
        if not downsample:
            self.downsample = IdentitySize()
        elif mixer[-1] == 'Conv':
            self.downsample = SubSample2D(dim, out_dim, sub_k)
        else:
            self.downsample = SubSample1D(dim, out_dim, sub_k)

    def forward(self, x, sz):
        return self.downsample(self.blocks(x), sz)


class ConvBNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class POPatchEmbed(nn.Module):
    """Two stride-2 convolutions: 1/4 of the height and width."""

    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.patch_embed = nn.Sequential(ConvBNLayer(in_channels, embed_dim // 2, 2),
                                         ConvBNLayer(embed_dim // 2, embed_dim, 2))

    def forward(self, x):
        return self.patch_embed(x), [x.shape[2] // 4, x.shape[3] // 4]


class SVTRv2(nn.Module):
    """SVTRv2 encoder (OpenOCR SVTRv2LNConvTwo33 with feat2d=True): [B, C, H, W] -> [B, dims[-1], H/16, W/4]
    for H = 32 * first_stride."""

    def __init__(self, in_channels=1, size='base', imgH=64, mlp_ratio=4, qkv_bias=True, drop_path_rate=0.1):
        super().__init__()
        if imgH % 32:
            raise ValueError(f'SVTRv2 needs --imgH to be a multiple of 32, got {imgH}')
        p = PRESETS[size]
        dims, depths, mixer = p['dims'], p['depths'], p['mixer']
        # a 32 px line keeps all 8 rows into stage 2 (as in OpenOCR); a taller line is reduced to them
        sub_k = [[imgH // 32, 1], [2, 1]]
        self.out_channels = dims[-1]
        self.pope = POPatchEmbed(in_channels, dims[0])
        dpr = np.linspace(0, drop_path_rate, sum(depths))
        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            last = i == len(depths) - 1
            self.stages.append(SVTRStage(dims[i], 0 if last else dims[i + 1], depths[i], mixer[i],
                                         None if last else sub_k[i], p['num_heads'][i], mlp_ratio, qkv_bias,
                                         dpr[sum(depths[:i]):sum(depths[:i + 1])], num_conv=2, downsample=not last))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0, std=0.02)
            if m.bias is not None:
                zeros_(m.bias)
        if isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        if isinstance(m, nn.Conv2d):
            kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x, sz = self.pope(x)
        for stage in self.stages:
            x, sz = stage(x, sz)
        return x.transpose(1, 2).reshape(-1, x.shape[-1], sz[0], sz[1])     # back to 2D


class RCTCHead(nn.Module):
    """OpenOCR's RCTCDecoder without the classifier: attention along each row, then a learned
    attention pooling over the height of every column (so a mark above or below the baseline is
    picked up where it is, not averaged away), then each column split into `upsample` frames.
    [B, C, H, W] -> [B, W * upsample, C]"""

    def __init__(self, in_channels, upsample=2):
        super().__init__()
        self.char_token = nn.Parameter(torch.zeros([1, 1, in_channels]))
        trunc_normal_(self.char_token, mean=0, std=0.02)
        self.fc_kv = nn.Linear(in_channels, 2 * in_channels)
        self.w_atten_block = PreNormBlock(in_channels, in_channels // 32)
        self.upsample = upsample
        if upsample > 1:
            self.split = nn.Linear(in_channels, in_channels * upsample)
            self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.w_atten_block(x.permute(0, 2, 3, 1).reshape(-1, W, C)).reshape(B, H, W, C).permute(0, 3, 1, 2)
        x_k, x_v = self.fc_kv(x.flatten(2).transpose(1, 2)).reshape(B, H * W, 2, C).permute(2, 0, 3, 1).unbind(0)
        attn = (self.char_token.expand(B, -1, -1) @ x_k).reshape(B, 1, H, W).softmax(2)    # over the height
        feats = (attn.permute(0, 3, 1, 2) @ x_v.reshape(B, C, H, W).permute(0, 3, 2, 1)).squeeze(2)  # [B, W, C]
        if self.upsample > 1:
            feats = self.norm(self.split(feats).reshape(B, W * self.upsample, C))
        return feats


def load_openocr(model, path, log=print):
    """Load the encoder and RCTC head of an OpenOCR SVTRv2 checkpoint into model, skipping the
    classifier and any layer whose shape differs (e.g. the RGB first convolution is summed to
    one grey channel)."""
    state = torch.load(path, map_location='cpu')
    state = state.get('state_dict', state)
    own = model.state_dict()
    loaded, skipped = 0, []
    for k, v in state.items():
        name = k.replace('module.', '', 1)
        if name.startswith('encoder.'):
            name = 'FeatureExtraction.' + name[len('encoder.'):]
        elif name.startswith('decoder.') and not name.startswith('decoder.fc.'):
            name = 'SequenceModeling.' + name[len('decoder.'):]
        else:
            skipped.append(k)
            continue
        if name in own and v.dim() == 4 and own[name].dim() == 4 and v.shape[1] == 3 and own[name].shape[1] == 1:
            v = v.sum(1, keepdim=True)
        if name in own and own[name].shape == v.shape:
            own[name] = v
            loaded += 1
        else:
            skipped.append(k)
    model.load_state_dict(own)
    log(f'{path}: loaded {loaded} tensors, skipped {len(skipped)} (classifier, or shape differs)')
