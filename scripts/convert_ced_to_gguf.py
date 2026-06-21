#!/usr/bin/env python3
"""Convert a CED (mispeech/ced-*) checkpoint to a self-contained GGUF.

The GGUF is fully metadata-driven: every dim/param lives in KV, the 527 AudioSet
labels are embedded, and the torchaudio mel filterbank + Hann window are baked in
as F32 buffers so the C++ engine never re-derives them (HTK mel + periodic Hann
are easy to get subtly wrong). Tensor names are kept **verbatim** from the
PyTorch ``state_dict`` so the C++ port is a 1:1 mapping. See docs/ARCHITECTURE.md.

Quantization (``--dtype f16|q8_0``) is applied ONLY to the linear weights the C++
engine feeds directly into ``ggml_mul_mat`` (attention qkv/proj, MLP fc1/fc2, and
the output Linear); ggml dequantizes those on the fly. Everything else — conv
patch-embed kernel, LayerNorm/BatchNorm params, positional embeddings, mel
filterbank/window — stays F32.

Usage:
    python scripts/convert_ced_to_gguf.py --model mispeech/ced-base \
        --output models/ced-base-f16.gguf --dtype f16
"""
import argparse
import re
import sys

import numpy as np
import torch
import torchaudio.transforms as AT
from transformers import AutoModelForAudioClassification

try:
    import gguf
except ImportError as e:  # pragma: no cover
    print(f"converter: missing dependency 'gguf': {e}", file=sys.stderr)
    print("CED_CONVERT_DEPS_MISSING", file=sys.stderr)
    sys.exit(2)


# Linear weights fed verbatim into ggml_mul_mat (safe to quantize). "N" is any
# layer index. Everything not matched here stays F32.
_QUANTIZABLE_PATTERNS = [
    r"^encoder\.blocks\.\d+\.attn\.qkv\.weight$",
    r"^encoder\.blocks\.\d+\.attn\.proj\.weight$",
    r"^encoder\.blocks\.\d+\.mlp\.fc1\.weight$",
    r"^encoder\.blocks\.\d+\.mlp\.fc2\.weight$",
    r"^outputlayer\.1\.weight$",
]
_QUANTIZABLE_RE = [re.compile(p) for p in _QUANTIZABLE_PATTERNS]


def should_quantize(name, ggml_ne, dtype):
    """ggml quant type for ``name`` (ggml_ne = reversed torch shape), or None=F32."""
    if dtype == "f32":
        return None
    if not any(rx.match(name) for rx in _QUANTIZABLE_RE):
        return None
    if len(ggml_ne) < 2 or ggml_ne[0] < 32 or ggml_ne[1] < 32:
        return None
    if dtype == "f16":
        return gguf.GGMLQuantizationType.F16
    if dtype == "q8_0":
        if ggml_ne[0] % 32 != 0:
            return None  # leading (contraction) dim not block-aligned -> keep F32
        return gguf.GGMLQuantizationType.Q8_0
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mispeech/ced-base", help="HF id or local path")
    ap.add_argument("--output", required=True)
    ap.add_argument("--dtype", choices=["f32", "f16", "q8_0"], default="f16")
    args = ap.parse_args()

    print(f"== loading {args.model}", file=sys.stderr)
    try:
        m = AutoModelForAudioClassification.from_pretrained(
            args.model, trust_remote_code=True
        ).eval()
    except Exception as e:  # pragma: no cover
        print(f"CED_MODEL_UNAVAILABLE: {e}", file=sys.stderr)
        sys.exit(2)
    cfg = m.config

    sr = 16000
    f_max = int(cfg.f_max) if getattr(cfg, "f_max", None) else sr // 2
    n_freqs = cfg.n_fft // 2 + 1

    # Build torchaudio's exact mel filterbank + window, then lift the buffers.
    melspec = AT.MelSpectrogram(
        f_min=cfg.f_min, sample_rate=sr, win_length=cfg.win_size, center=cfg.center,
        n_fft=cfg.n_fft, f_max=(f_max if getattr(cfg, "f_max", None) else None),
        hop_length=cfg.hop_size, n_mels=cfg.n_mels,
    )
    fb = melspec.mel_scale.fb.detach().cpu().float().numpy()          # [n_freqs, n_mels]
    window = melspec.spectrogram.window.detach().cpu().float().numpy()  # [win_size]
    assert fb.shape == (n_freqs, cfg.n_mels), fb.shape
    # store mel-major [n_mels, n_freqs] so C++ does mel = fb @ power directly
    fb_mel_major = np.ascontiguousarray(fb.T, dtype=np.float32)       # [n_mels, n_freqs]

    w = gguf.GGUFWriter(args.output, "ced")
    w.add_string("general.name", args.model)
    w.add_string("ced.arch", "ced")

    # transformer dims
    w.add_uint32("ced.embed_dim", int(cfg.embed_dim))
    w.add_uint32("ced.depth", int(cfg.depth))
    w.add_uint32("ced.num_heads", int(cfg.num_heads))
    w.add_float32("ced.mlp_ratio", float(cfg.mlp_ratio))
    w.add_uint32("ced.outputdim", int(cfg.outputdim))
    w.add_bool("ced.qkv_bias", bool(getattr(cfg, "qkv_bias", True)))
    w.add_uint32("ced.patch_size", int(cfg.patch_size))
    w.add_uint32("ced.patch_stride", int(cfg.patch_stride))
    w.add_uint32("ced.target_length", int(cfg.target_length))
    w.add_string("ced.pooling", str(cfg.pooling))

    # norm epsilons (encoder norms use 1e-6; the head LayerNorm uses torch default
    # 1e-5; init_bn is BatchNorm2d torch default 1e-5).
    w.add_float32("ced.ln_eps_encoder", 1e-6)
    w.add_float32("ced.ln_eps_head", 1e-5)
    w.add_float32("ced.bn_eps", 1e-5)

    # mel frontend
    w.add_uint32("ced.sample_rate", sr)
    w.add_uint32("ced.n_mels", int(cfg.n_mels))
    w.add_uint32("ced.n_fft", int(cfg.n_fft))
    w.add_uint32("ced.win_size", int(cfg.win_size))
    w.add_uint32("ced.hop_size", int(cfg.hop_size))
    w.add_uint32("ced.n_freqs", int(n_freqs))
    w.add_float32("ced.f_min", float(cfg.f_min))
    w.add_float32("ced.f_max", float(f_max))
    w.add_bool("ced.center", bool(cfg.center))
    # AmplitudeToDB(stype="power", top_db=120): db = 10*log10(max(x,amin)); ref=1
    w.add_float32("ced.a2db_multiplier", 10.0)
    w.add_float32("ced.a2db_amin", 1e-10)
    w.add_float32("ced.a2db_top_db", 120.0)
    w.add_float32("ced.a2db_ref", 1.0)

    # 527 AudioSet labels (index order)
    labels = [str(cfg.id2label[i]) for i in range(int(cfg.outputdim))]
    w.add_array("ced.labels", labels)

    # baked frontend buffers
    w.add_tensor("ced.mel_filterbank", fb_mel_major)   # [n_mels, n_freqs]
    w.add_tensor("ced.mel_window", np.ascontiguousarray(window, dtype=np.float32))

    # weights (verbatim names), allowlisted linears quantized per --dtype
    sd = m.state_dict()
    written = quantized = 0
    for name, t in sd.items():
        if not hasattr(t, "detach"):
            continue
        arr = t.detach().cpu().float().numpy()
        if arr.ndim == 0:
            continue  # skip scalar bookkeeping (num_batches_tracked)
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        ggml_ne = list(arr.shape[::-1])
        qtype = should_quantize(name, ggml_ne, args.dtype)
        if qtype is None:
            w.add_tensor(name, arr)
        else:
            raw = gguf.quantize(arr, qtype)
            w.add_tensor(name, raw, raw_shape=raw.shape, raw_dtype=qtype)
            quantized += 1
        written += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(
        f"== wrote {args.output}: arch=ced embed_dim={cfg.embed_dim} depth={cfg.depth} "
        f"labels={len(labels)} tensors={written} dtype={args.dtype} quantized={quantized}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
