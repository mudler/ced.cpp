#!/usr/bin/env python3
"""Dump CED reference intermediate tensors as a baseline GGUF (gold parity targets).

Runs the real ``mispeech/ced-*`` model on a fixed, deterministic test waveform and
captures every intermediate tensor in forward order. The C++/ggml port is gated
against these per-component (numerically equal within tolerance) before the next
component is built. See docs/ARCHITECTURE.md.

The manual forward here mirrors modeling_ced.py exactly; we assert the final probs
match ``model(**inputs).logits`` so the dump itself is verified against the model.

Usage:
    python scripts/gen_ced_baseline.py --model mispeech/ced-base --out tests/fixtures/ced-base.baseline.gguf
"""
import argparse
import sys

import numpy as np
import torch
import torchaudio.transforms as AT
from transformers import AutoModelForAudioClassification

try:
    import gguf
except ImportError as e:  # pragma: no cover
    print(f"baseline: missing dependency 'gguf': {e}", file=sys.stderr)
    sys.exit(2)


def make_waveform(n_samples: int) -> torch.Tensor:
    """Deterministic, mel-exciting test signal: two tones + a little noise."""
    g = torch.Generator().manual_seed(1234)
    t = torch.arange(n_samples, dtype=torch.float32) / 16000.0
    wav = (
        0.30 * torch.sin(2 * np.pi * 440.0 * t)
        + 0.20 * torch.sin(2 * np.pi * 1000.0 * t)
        + 0.05 * torch.randn(n_samples, generator=g)
    )
    return wav.float()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mispeech/ced-base")
    ap.add_argument("--out", default="tests/fixtures/ced-base.baseline.gguf")
    # T = 1 + floor(n_samples/hop); 161760/160 = 1011 -> T = 1012 = target_length
    ap.add_argument("--n-samples", type=int, default=161760)
    args = ap.parse_args()

    torch.manual_seed(0)
    print(f"== loading {args.model}", file=sys.stderr)
    model = AutoModelForAudioClassification.from_pretrained(
        args.model, trust_remote_code=True
    ).eval()
    cfg = model.config
    enc = model.encoder

    wav = make_waveform(args.n_samples)

    # --- frontend (torchaudio, exactly as feature_extraction_ced.py) ---
    f_max = cfg.f_max if getattr(cfg, "f_max", None) else None
    melspec = AT.MelSpectrogram(
        f_min=cfg.f_min,
        sample_rate=16000,
        win_length=cfg.win_size,
        center=cfg.center,
        n_fft=cfg.n_fft,
        f_max=f_max,
        hop_length=cfg.hop_size,
        n_mels=cfg.n_mels,
    )
    a2db = AT.AmplitudeToDB(stype="power", top_db=120)

    dumps: dict[str, np.ndarray] = {}

    def dump(name: str, t: torch.Tensor):
        dumps[name] = t.detach().contiguous().float().numpy().astype(np.float32)

    with torch.no_grad():
        mel_power = melspec(wav.unsqueeze(0))      # [1,64,T]
        input_values = a2db(mel_power)             # [1,64,T]
        T = input_values.shape[-1]
        # Single-chunk path (what the C++ gates against): T must not exceed the
        # positional grid. T < target_length is allowed (short clip) and
        # exercises the sliced time_pos_embed path.
        assert T <= cfg.target_length, f"T={T} > target_length={cfg.target_length}"

        dump("audio_waveform", wav)
        dump("mel_power", mel_power.squeeze(0))
        dump("input_values", input_values.squeeze(0))

        # --- encoder: init_bn (CedModel.forward layout dance) ---
        x = input_values.unsqueeze(1)              # [1,1,64,T]
        x = x.permute(0, 2, 1, 3)                  # [1,64,1,T]
        x = enc.init_bn(x)
        x = x.permute(0, 2, 1, 3)                  # [1,1,64,T]
        dump("init_bn_out", x.squeeze(0).squeeze(0))   # [64,T]

        # --- forward_features ---
        x = enc.patch_embed(x)                     # [1,768,4,63] (flatten=False)
        dump("patch_embed", x.squeeze(0))          # [768,4,63]
        tt = x.shape[-1]
        x = x + enc.time_pos_embed[:, :, :, :tt]
        x = x + enc.freq_pos_embed[:, :, :, :]
        dump("pos_out", x.squeeze(0))              # [768,4,63]
        x = torch.permute(torch.flatten(x, 2, 3), (0, 2, 1))  # [1,252,768]
        dump("tokens_in", x.squeeze(0))            # [252,768]
        x = enc.pos_drop(x)
        for i, blk in enumerate(enc.blocks):
            x = blk(x)
            dump(f"block_{i}", x.squeeze(0))        # [252,768]
        x = enc.norm(x)
        dump("enc_norm", x.squeeze(0))             # [252,768]

        # --- head (forward_head, pooling="mean") ---
        pooled = x.mean(1)                          # [1,768]
        dump("pooled", pooled.squeeze(0))
        ln = model.outputlayer[0](pooled)
        logits = model.outputlayer[1](ln)           # [1,527]
        dump("logits", logits.squeeze(0))
        probs = torch.sigmoid(logits)
        dump("probs", probs.squeeze(0))

        # --- verify the manual forward against the model itself ---
        ref = model(input_values=input_values).logits  # forward_head returns sigmoid
        max_d = (ref - probs).abs().max().item()
        print(f"== manual-vs-model max|d| (probs) = {max_d:.3e}", file=sys.stderr)
        assert max_d < 1e-5, f"manual forward diverges from model: {max_d}"

    # --- write baseline GGUF ---
    w = gguf.GGUFWriter(args.out, arch="ced-baseline")
    w.add_uint32("ced.target_length", int(cfg.target_length))
    w.add_uint32("ced.n_mels", int(cfg.n_mels))
    w.add_uint32("ced.outputdim", int(cfg.outputdim))
    w.add_uint32("ced.embed_dim", int(cfg.embed_dim))
    w.add_uint32("ced.depth", int(cfg.depth))
    for name, arr in dumps.items():
        w.add_tensor(name, np.ascontiguousarray(arr))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"== wrote {args.out}  ({len(dumps)} tensors)", file=sys.stderr)
    for n in ("input_values", "init_bn_out", "patch_embed", "tokens_in",
              "block_0", "block_11", "enc_norm", "pooled", "logits", "probs"):
        print(f"   {n:14s} {dumps[n].shape}", file=sys.stderr)

    # top-5 predicted tags, as a sanity check of what the test clip "sounds like"
    top = np.argsort(-dumps["probs"])[:5]
    print("== top-5 tags:", file=sys.stderr)
    for i in top:
        print(f"   {dumps['probs'][i]:.3f}  {cfg.id2label[int(i)]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
