---
license: apache-2.0
library_name: ced.cpp
pipeline_tag: audio-classification
tags:
  - audio-classification
  - sound-event-detection
  - audio-tagging
  - audioset
  - ggml
  - gguf
  - ced
base_model:
  - mispeech/ced-tiny
  - mispeech/ced-mini
  - mispeech/ced-small
  - mispeech/ced-base
---

# CED (GGUF) for ced.cpp / LocalAI

GGUF quantizations of the **CED** family (Consistent Ensemble Distillation,
Xiaomi) - SOTA-tier audio-tagging models that classify everyday sounds (baby
cry, footsteps, glass breaking, alarms, dog bark, ...) into the 527-class
[AudioSet](https://research.google.com/audioset/) ontology.

These files run with [**ced.cpp**](https://github.com/mudler/ced.cpp), a
standalone C++/[ggml](https://github.com/ggml-org/ggml) port (no Python, no
PyTorch at inference), and with [**LocalAI**](https://github.com/mudler/LocalAI)
via the `ced` backend. Converted from the `mispeech/ced-*` checkpoints
(Apache-2.0). CED is a plain AST/DeiT Vision Transformer over a log-mel
spectrogram; the port is numerically equal to the PyTorch reference.

## Files

One self-contained GGUF per size + quant (config, 527 labels, and the mel
filterbank/window are all embedded). Pick by your accuracy/size budget:

| size | params | f16 | q8_0 | f32 |
|------|--------|-----|------|-----|
| **tiny**  | 5.5M | `ced-tiny-f16.gguf` (11 MB)  | `ced-tiny-q8_0.gguf` (6 MB)  | - |
| **mini**  | 9.6M | `ced-mini-f16.gguf` (19 MB)  | `ced-mini-q8_0.gguf` (11 MB) | - |
| **small** | 22M  | `ced-small-f16.gguf` (42 MB) | `ced-small-q8_0.gguf` (23 MB)| - |
| **base**  | 86M  | `ced-base-f16.gguf` (165 MB) | `ced-base-q8_0.gguf` (88 MB) | `ced-base-f32.gguf` (328 MB) |

`tiny`/`q8_0` (6 MB) is ideal for Raspberry-Pi-class CPUs; `base`/`f16` is the
accuracy default.

## Parity vs PyTorch (ced-base, end-to-end probs)

| quant | max abs diff | top-5 tags |
|-------|--------------|------------|
| f32  | 1.7e-7 | identical |
| f16  | 6.4e-5 | identical |
| q8_0 | 6.0e-3 | identical |

## Performance (CPU, ced-base, 10s clip, Ryzen 9 9950X3D, 4 threads)

| | latency | realtime factor | peak RSS |
|---|---|---|---|
| PyTorch (transformers, f32) | 155.7 ms | 65x | 717 MB |
| ced.cpp f16  | 100.6 ms | 100x | 189 MB |
| ced.cpp q8_0 | 117.2 ms |  86x | 111 MB |

ced.cpp f16 is ~1.55x faster than the PyTorch reference; q8_0 uses ~6.5x less
memory.

## Usage

```sh
ced-cli classify ced-base-f16.gguf clip.wav --top-k 5
# 0.87  Baby cry, infant cry
# 0.12  Crying, sobbing
```

In LocalAI: install the `ced` backend, configure a model with one of these
GGUFs, then call `POST /v1/audio/classification` (or stream over the realtime
websocket API for live recognition).

## License

Model weights: **Apache-2.0** (© Xiaomi Corporation; from the `mispeech/ced-*`
checkpoints). AudioSet labels are CC-BY-4.0. The ced.cpp inference code is MIT.
