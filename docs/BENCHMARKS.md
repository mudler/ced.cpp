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
| PyTorch (transformers, f32) | 158.8 ms | 64x | 399.1 ms | 717 MB |
| **ced.cpp f32** (same precision) | 126.6 ms | 80x  | 433.3 ms | 354 MB |
| **ced.cpp f16**       | **102.9 ms** | **98x** | 354.3 ms | 189 MB |
| **ced.cpp q8_0**      | 117.1 ms | 86x  | 410.3 ms | **111 MB** |

RTF = clip seconds / inference seconds (higher is faster; ~100x = 100 s of audio
classified per wall-second).

## Takeaways

- **Same precision (f32 vs f32)** is the apples-to-apples comparison: ced.cpp is
  **~1.25x faster (126.6 vs 158.8 ms) and uses 2x less memory (354 vs 717 MB)**
  than PyTorch, at the same numerical output.
- **The quantized configs go further** (near-lossless: identical top-5 tags):
  **f16 is the CPU sweet spot** at 102.9 ms/clip (~1.5x faster than PyTorch f32,
  ~100x realtime - tinyBLAS's f16 GEMM is the win), and **q8_0 drops to 111 MB**
  (~6.5x less than PyTorch) for a small dequant-overhead latency cost.
- **No Python/torch runtime is resident**, so peak RAM is much lower across the board.
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
