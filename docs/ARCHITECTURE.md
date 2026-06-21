# CED architecture (ced-base) — port reference

Source of truth: `mispeech/ced-base` on HuggingFace (Apache-2.0 **weights**;
upstream training code at `RicherMans/CED` is GPL-3.0 — **not** used here. This
port is a clean-room reimplementation of the inference forward pass from the
architecture below, consuming only the Apache-2.0 weights).

CED ("Consistent Ensemble Distillation", Xiaomi) is a plain AST/DeiT-style audio
Vision Transformer over a log-mel spectrogram, producing 527 AudioSet tags. The
inference graph has **no custom ops** — it is LayerNorm + MHSA + GELU MLP +
conv2d patchify + two broadcast positional adds.

## Model sizes (one codebase covers all)

| variant | embed_dim | depth | heads | params |
|---|---|---|---|---|
| ced-tiny  | 192 | 12 | 3  | 5.5M |
| ced-mini  | 256 | 12 | 4  | 9.6M |
| ced-small | 384 | 12 | 6  | 22M  |
| ced-base  | 768 | 12 | 12 | 86M  |

All dims are read from the GGUF; nothing is hardcoded. Numbers below are ced-base.

## Frontend (feature extractor — torchaudio)

Input: mono waveform, 16 kHz, float32 in roughly [-1, 1].

1. `torchaudio.transforms.MelSpectrogram`:
   - `sample_rate=16000`, `n_fft=512`, `win_length=512`, `hop_length=160`
   - `f_min=0`, `f_max=None` → defaults to `sample_rate/2 = 8000`
   - `n_mels=64`, `center=True` (reflect padding), `power=2.0`
   - mel filterbank defaults: `norm=None`, `mel_scale="htk"` (NOT librosa/slaney)
   - window: Hann (`torch.hann_window`, periodic) of length 512
   - output shape `(64, T)` where `T = 1 + floor(n_samples / 160)` with center pad
2. `torchaudio.transforms.AmplitudeToDB(stype="power", top_db=120)`:
   - `x_db = 10 * log10(clamp(x, min=1e-10))`  (ref=1.0 → no offset)
   - `x_db = max(x_db, x_db.amax(per-spectrogram) - 120)`
   - result is `input_values`, shape `(64, T)`

**Parity strategy:** the converter bakes torchaudio's exact mel filterbank
`[64, 257]` and Hann window `[512]` into the GGUF as F32 buffers, so the C++
side never re-derives them (HTK mel + periodic Hann are easy to get subtly
wrong). C++ does: reflect-pad → framed rFFT (n_fft=512, hop=160) → power
spectrum `[257, T]` → `filterbank @ power` → `[64, T]` → AmplitudeToDB.

`target_length = 1012` frames (~10.1 s). Clips longer than 1012 frames are split
into 1012-frame chunks (last padded if `pad_last`), each chunked clip encoded
independently and the per-chunk logits stacked (caller averages). For the
baseline we use exactly 1012 frames so `n_splits = 1`.

## Encoder (`CedModel`)

Tensor names are kept **verbatim** from the PyTorch `state_dict`.

1. **init_bn** — `BatchNorm2d(64)` applied over the mel dimension (eval mode):
   `y = (x - running_mean) / sqrt(running_var + eps) * weight + bias`, per-mel,
   `eps = 1e-5`. Tensors: `encoder.init_bn.{weight,bias,running_mean,running_var}`.
   Fold into a per-mel scale+bias at convert time.
   - Layout dance in PyTorch: `(B,64,T) → unsqueeze(1) → (B,1,64,T) →
     permute(0,2,1,3) → (B,64,1,T) → init_bn → permute back → (B,1,64,T)`.
     Net effect: normalize each of the 64 mel rows by its BN stats.

2. **patch_embed** — `Conv2d(1, 768, kernel=16, stride=16)` on `(B,1,64,1012)`
   → `(B, 768, F=4, Tg=63)` (`F = 64/16`, `Tg = floor(1012/16) = 63`).
   Tensors: `encoder.patch_embed.proj.{weight[768,1,16,16],bias[768]}`.

3. **positional** (factorized, added on the `(B,768,4,63)` grid):
   - `x += encoder.time_pos_embed[:, :, :, :63]`  shape `(1,768,1,63)`, broadcast over freq
   - `x += encoder.freq_pos_embed`                shape `(1,768,4,1)`, broadcast over time

4. **flatten to tokens** — `permute(flatten(x, 2, 3), (0,2,1))`:
   `(B,768,4,63) → (B,768,252) → (B,252,768)`. Flatten is **freq-major**:
   token index `= f*63 + t` (f in 0..3 outer, t in 0..62 inner). N = 252 tokens.

5. **12 × CedBlock** (pre-norm, residual):
   ```
   x = x + attn(norm1(x))
   x = x + mlp (norm2(x))
   ```
   - `norm1`, `norm2`: LayerNorm, **eps = 1e-6**.
   - `attn`: `qkv = Linear(768, 2304)`; split to q,k,v of 12 heads × 64;
     `attn = softmax((q @ kᵀ) * scale)`, `scale = 64^-0.5 = 0.125`; no mask
     (`causal=False`); `out = Linear(768,768)(attn @ v)`.
     Tensors: `...attn.qkv.{weight,bias}`, `...attn.proj.{weight,bias}`.
   - `mlp`: `fc2(GELU(fc1(x)))`, hidden 3072. **GELU is exact erf**
     (`nn.GELU()`, approximate='none') → use `ggml_gelu_erf`, NOT the tanh
     approximation. Tensors: `...mlp.fc1.{weight,bias}`, `...mlp.fc2.{weight,bias}`.
   - `ls1`/`ls2` and `drop_path` are Identity at inference.

6. **final norm** — `encoder.norm` LayerNorm, **eps = 1e-6**. Output `(B,252,768)`.

## Head (`CedForAudioClassification.forward_head`, pooling="mean")

1. `x = x.mean(1)` — mean over the 252 tokens → `(B,768)`.
2. `outputlayer`:
   - `outputlayer.0`: LayerNorm(768), **default eps = 1e-5** (NOT 1e-6 — this LN
     is constructed plainly, unlike the encoder norms).
   - `outputlayer.1`: Linear(768, 527). Tensors `outputlayer.1.{weight[527,768],bias[527]}`.
3. `probs = sigmoid(logits)` — multi-label, per-class independent probabilities.

`config.pooling` is `"mean"` for the released tagger checkpoints. (Other modes —
`token`, `logit`, `dm` — exist in the reference but are not used by ced-* tags.)

## Parity gate points (dumped by gen_ced_baseline.py)

In forward order, each dumped as an F32 tensor in the baseline GGUF:
- `mel_power`     `[257? no → 64, T]`  (post-MelSpectrogram, pre-dB) — optional
- `input_values`  `[64, T]`            (post-AmplitudeToDB)
- `init_bn_out`   `[64, T]`
- `patch_embed`   `[768, 4, 63]`       (post conv2d, pre-pos)
- `pos_out`       `[768, 4, 63]`       (post both positional adds)
- `tokens_in`     `[252, 768]`         (post flatten, pre-blocks)
- `block_{0..11}` `[252, 768]`         (output of each CedBlock)
- `enc_norm`      `[252, 768]`         (post final LayerNorm)
- `pooled`        `[768]`              (mean over tokens)
- `logits`        `[527]`              (pre-sigmoid)
- `probs`         `[527]`              (post-sigmoid)

C++ must match each within tight tolerance (f32: ~1e-4 atol) before the next
component is built. End-to-end text/argmax is NOT a sufficient gate.

## GGUF KV schema (metadata-driven)

`ced.arch="ced"`, `ced.embed_dim`, `ced.depth`, `ced.num_heads`, `ced.mlp_ratio`,
`ced.outputdim`, `ced.n_mels`, `ced.n_fft`, `ced.win_size`, `ced.hop_size`,
`ced.f_min`, `ced.f_max`, `ced.sample_rate`, `ced.center`, `ced.target_length`,
`ced.patch_size`, `ced.patch_stride`, `ced.pooling`, `ced.ln_eps_encoder=1e-6`,
`ced.ln_eps_head=1e-5`, `ced.bn_eps=1e-5`, plus `ced.labels` (527 strings) and
the baked `mel_filterbank` / `mel_window` buffers.
