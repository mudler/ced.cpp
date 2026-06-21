#!/usr/bin/env python3
"""Benchmark the reference PyTorch CED (mispeech/ced-*) end-to-end on CPU.

Times the full clip -> 527 probs path (torchaudio mel + ViT forward), matching
what ced.cpp's `ced-cli bench` measures, so the two are directly comparable.

Usage:
    python scripts/bench_torch.py --model mispeech/ced-base --wav /tmp/ced_test.wav \
        --iters 30 --warmup 5 --threads 4
"""
import argparse
import statistics
import sys
import time

import numpy as np
import torch
import torchaudio.transforms as AT
from transformers import AutoModelForAudioClassification


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mispeech/ced-base")
    ap.add_argument("--wav", default="/tmp/ced_test.wav")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)

    import soundfile as sf
    wav_np, sr = sf.read(args.wav, dtype="float32")
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=1)
    clip_s = len(wav_np) / sr
    wav = torch.from_numpy(np.ascontiguousarray(wav_np)).float()

    model = AutoModelForAudioClassification.from_pretrained(
        args.model, trust_remote_code=True
    ).eval()
    cfg = model.config
    f_max = cfg.f_max if getattr(cfg, "f_max", None) else None
    melspec = AT.MelSpectrogram(
        f_min=cfg.f_min, sample_rate=16000, win_length=cfg.win_size, center=cfg.center,
        n_fft=cfg.n_fft, f_max=f_max, hop_length=cfg.hop_size, n_mels=cfg.n_mels,
    )
    a2db = AT.AmplitudeToDB(stype="power", top_db=120)

    def run():
        # full end-to-end: mel frontend + model forward (-> sigmoid probs)
        iv = a2db(melspec(wav.unsqueeze(0)))
        return model(input_values=iv).logits

    for _ in range(args.warmup):
        run()

    ms = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        run()
        ms.append((time.perf_counter() - t0) * 1000.0)
    ms.sort()
    mean = statistics.fmean(ms)
    median = ms[len(ms) // 2]

    print(f"model={args.model}  clip={clip_s:.2f}s  threads={args.threads}  iters={args.iters}",
          file=sys.stderr)
    print(f"  latency ms: min={ms[0]:.2f}  median={median:.2f}  mean={mean:.2f}  max={ms[-1]:.2f}",
          file=sys.stderr)
    print(f"  RTF (clip_s/mean_s): {clip_s / (mean / 1000.0):.1f}x realtime  "
          f"({1000.0 / mean:.1f} clips/s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
