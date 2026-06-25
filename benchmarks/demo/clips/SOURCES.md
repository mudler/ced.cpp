# Demo clip sources

All three demo clips are **Public Domain** audio from Wikimedia Commons, trimmed
to a clean 6.0 s window and converted to 16 kHz mono WAV (the format ced.cpp
consumes). No attribution is legally required for public-domain works; credit is
listed below anyway.

| file | source (Wikimedia Commons) | author | license | trim |
|------|----------------------------|--------|---------|------|
| `rooster.wav` | [Medium rooster crowing.ogg](https://commons.wikimedia.org/wiki/File:Medium_rooster_crowing.ogg) | alys | Public domain | first 6.0 s |
| `thunder.wav` | [Rain and thunder.ogg](https://commons.wikimedia.org/wiki/File:Rain_and_thunder.ogg) | (Wikimedia Commons) | Public domain | first 6.0 s |
| `guitar.wav` | [AcousticGuitarSample.ogg](https://commons.wikimedia.org/wiki/File:AcousticGuitarSample.ogg) | RyGuy | Public domain | 6.0 s from t=6 s |

Reconvert from an original `.ogg` with:

```sh
ffmpeg -ss <start> -t 6 -i original.ogg -ac 1 -ar 16000 clip.wav
```

The tags and latencies shown in the demo come from running these exact WAVs
through `ced-cli classify` / `ced-cli bench` (see `spec.json` for the captured
values and the box they were measured on).
