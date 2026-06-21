#!/usr/bin/env python3
"""Diverse-input numerical parity: ced.cpp C-API vs the upstream PyTorch model.

Drives the SHIPPED C-API (libced.so, via ctypes) and the upstream HF model
(mispeech/ced-*) on several distinct signals - white noise, near-silence, low/
high tones, a short chirp - and asserts the 527-class probabilities match and
the top-1 label agrees. Complements the per-component baseline gates (which use
a single synthetic clip) by exercising varied spectra + the short-clip path.

Usage:
    python scripts/parity_diverse.py --lib build-shared/libced.so \
        --model models/ced-base-f32.gguf --hf mispeech/ced-base
"""
import argparse
import ctypes
import json
import sys

import numpy as np
import torch
import torchaudio.transforms as AT
from transformers import AutoModelForAudioClassification


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="build-shared/libced.so")
    ap.add_argument("--model", default="models/ced-base-f32.gguf")
    ap.add_argument("--hf", default="mispeech/ced-base")
    ap.add_argument("--atol", type=float, default=2e-3)
    args = ap.parse_args()

    m = AutoModelForAudioClassification.from_pretrained(args.hf, trust_remote_code=True).eval()
    cfg = m.config
    mel = AT.MelSpectrogram(f_min=cfg.f_min, sample_rate=16000, win_length=cfg.win_size,
                            center=cfg.center, n_fft=cfg.n_fft, f_max=cfg.f_max,
                            hop_length=cfg.hop_size, n_mels=cfg.n_mels)
    a2db = AT.AmplitudeToDB(stype="power", top_db=120)

    def torch_probs(wav):
        with torch.no_grad():
            return m(input_values=a2db(mel(torch.from_numpy(wav).float().unsqueeze(0)))).logits.squeeze(0).numpy()

    lib = ctypes.CDLL(args.lib)
    lib.ced_capi_load.restype = ctypes.c_void_p
    lib.ced_capi_load.argtypes = [ctypes.c_char_p]
    lib.ced_capi_classify_pcm_json.restype = ctypes.c_void_p
    lib.ced_capi_classify_pcm_json.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                               ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.ced_capi_free_string.argtypes = [ctypes.c_void_p]
    ctx = lib.ced_capi_load(args.model.encode())
    if not ctx:
        print("failed to load model", file=sys.stderr)
        return 1

    def ced_probs(wav):
        arr = np.ascontiguousarray(wav, dtype=np.float32)
        p = lib.ced_capi_classify_pcm_json(ctx, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                           len(arr), 16000, cfg.outputdim)
        s = ctypes.string_at(p).decode()
        lib.ced_capi_free_string(p)
        out = np.zeros(cfg.outputdim, dtype=np.float32)
        for d in json.loads(s):
            out[d["index"]] = d["score"]
        return out

    g = np.random.default_rng(0)
    sr, n = 16000, 16000 * 5
    t = np.arange(n) / sr
    sigs = {
        "white_noise":  (0.3 * g.standard_normal(n)).astype("float32"),
        "near_silence": (1e-4 * g.standard_normal(n)).astype("float32"),
        "low_100hz":    (0.5 * np.sin(2 * np.pi * 100 * t)).astype("float32"),
        "high_7khz":    (0.5 * np.sin(2 * np.pi * 7000 * t)).astype("float32"),
        "chirp_0.5s":   (0.4 * np.sin(2 * np.pi * (200 + 1500 * t) * t)).astype("float32")[: sr // 2],
    }

    print(f"{'signal':14s} {'max|d|':>10s} {'top1':>6s}  label")
    ok = True
    for name, w in sigs.items():
        tp, cp = torch_probs(w), ced_probs(w)
        md = float(np.max(np.abs(tp - cp)))
        t1, c1 = int(np.argmax(tp)), int(np.argmax(cp))
        match = t1 == c1
        ok &= md < args.atol and match
        print(f"{name:14s} {md:10.2e} {str(match):>6s}  {cfg.id2label[t1]}")
    print("ALL OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
