# Benchmarks: ced.cpp vs PyTorch reference

End-to-end per-clip latency (mel frontend + ViT encoder + head → 527 probs),
CPU-only, same machine, same clip. ced.cpp matches the PyTorch reference
numerically (see [parity](../README.md)); this measures speed and memory.

- **Machine**: AMD Ryzen 9 9950X3D, CPU only (no GPU).
- **Model**: ced-base (86M params).
- **Clip**: 10.11 s (1012 mel frames) — the single-chunk path.
- **PyTorch**: `transformers` + `torchaudio`, f32 (no native CPU f16/int8).
- **ced.cpp**: ggml CPU, `-march=native` + tinyBLAS, built `Release`.
- 40 timed iterations after 8 warmup; mean reported. Reproduce with the commands
  at the bottom.

## Latency

| Implementation        | 4-thread mean | 4-thread RTF | 1-thread mean | peak RSS |
|-----------------------|--------------:|-------------:|--------------:|---------:|
| PyTorch (transformers, f32) | 155.7 ms | 65x | 399.1 ms | 717 MB |
| **ced.cpp f32**       | 124.7 ms | 81x  | 433.3 ms | 354 MB |
| **ced.cpp f16**       | **100.6 ms** | **100x** | 354.3 ms | 189 MB |
| **ced.cpp q8_0**      | 117.2 ms | 86x  | 410.3 ms | **111 MB** |

RTF = clip seconds / inference seconds (higher is faster; 100x = 100 s of audio
classified per wall-second).

## Takeaways

- **f16 is the CPU sweet spot**: 100.6 ms/clip, **1.55x faster than PyTorch** at
  4 threads, and ~100x realtime. tinyBLAS's f16 GEMM is the win; q8_0 is slightly
  slower (dequant overhead) but the smallest footprint.
- **Memory**: ced.cpp q8_0 uses **111 MB vs PyTorch's 717 MB (6.5x less)**; f16 is
  3.8x less. No Python/torch runtime is resident.
- **Speedup at 4 threads vs PyTorch**: f16 1.55x, q8_0 1.33x, f32 1.25x.
- **Single thread**: ced.cpp f16/q8 still beat PyTorch, but f32 is marginally
  slower (PyTorch's oneDNN GEMM is strong single-threaded). On CPU, prefer f16.
- **Cold start**: `ced-cli classify` (model load + inference) completes in ~0.15 s
  wall; the PyTorch path pays multi-second `import torch`/`transformers` startup
  before the first inference — a large practical gap for serving and CLI use.
- **Parity holds throughout**: identical top-5 tags across all variants.

## Reproduce

```sh
# ced.cpp (per quant level, 4 and 1 threads)
ced-cli bench models/ced-base-f16.gguf  /tmp/ced_test.wav --iters 40 --warmup 8 --threads 4
ced-cli bench models/ced-base-q8_0.gguf /tmp/ced_test.wav --iters 40 --warmup 8 --threads 1

# PyTorch reference (same clip + methodology)
python scripts/bench_torch.py --model mispeech/ced-base --wav /tmp/ced_test.wav \
    --iters 40 --warmup 8 --threads 4
```
