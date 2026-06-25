#!/usr/bin/env python3
r"""ced_reel - the ced.cpp CAPABILITY REEL: a silent-readable demo that shows what
the model DOES rather than how fast it is. Three real Public-Domain sounds play in
turn (a rooster, a thunderstorm, an acoustic guitar); for each one a mel-spectrogram (the
representation the model actually consumes) wipes in under a playhead, then the
top-5 AudioSet tags snap in with confidence bars filling to their REAL measured
values, with a "tagged in N ms on CPU" stat. It ends on the LocalAI card.

Every tag, confidence and millisecond on screen is a real ced-cli output captured
in spec.json (ced-cli classify --top-k 5 for the tags, ced-cli bench for the
latency). The clips in clips/ are the exact 16 kHz mono WAVs that produced those
numbers. No invented numbers, no em-dashes.

Pure Pillow + numpy frame synthesis + ffmpeg palettegen/paletteuse, the same house
treatment as the other LocalAI .cpp demos (locate-anything.cpp/image_race,
voice-detect.cpp/voice_race). Renders a 16:9 hero and a 1:1 square cut in one pass:

  python3 ced_reel.py                       # reads ./spec.json, writes ./out
  python3 ced_reel.py --beat 3.4 --fps 24   # slower beats, smoother

Outputs out/ced_reel.{mp4,gif} (1280x720) and out/ced_reel_square.{mp4,gif}
(1080x1080). --beat sets seconds per sound; it is playback only and never changes
the displayed tags/latency, which are the real measured values from spec.json.
"""
import argparse, json, subprocess, tempfile, wave
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent

BG, PANEL, INK, DIM = (13, 17, 23), (22, 28, 36), (215, 221, 229), (120, 128, 139)
DIMMER = (96, 105, 117)
GRID = (34, 43, 52)
TEAL = (62, 200, 224)
TEAL_DIM = (44, 120, 138)
SLATE = (150, 165, 180)
GREEN = (102, 214, 130)
GOLD = (240, 205, 120)


# ----------------------------------------------------------------------------- fonts
def fontp(bold):
    return f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"


def font(sz, bold=True):
    try:
        return ImageFont.truetype(fontp(bold), sz)
    except Exception:
        return ImageFont.load_default()


def fmono(sz, bold=False):
    try:
        return ImageFont.truetype(
            f"/usr/share/fonts/truetype/dejavu/DejaVuSansMono{'-Bold' if bold else ''}.ttf", sz)
    except Exception:
        return font(sz, bold)


def ease(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def lerp(c0, c1, f):
    f = max(0.0, min(1.0, f))
    return tuple(int(c0[k] + (c1[k] - c0[k]) * f) for k in range(3))


def rounded(d, rect, r, fill=None, outline=None, width=1):
    d.rounded_rectangle(rect, r, fill=fill, outline=outline, width=width)


# ----------------------------------------------------------------------- mel spectrogram
def _read_wav(path):
    w = wave.open(str(path), "rb")
    sr = w.getframerate()
    raw = w.readframes(w.getnframes())
    w.close()
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return a, sr


def _mel_fb(n_mels, n_fft, sr, fmin=50.0, fmax=8000.0):
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mpts = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    hz = mel2hz(mpts)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        if c > l:
            fb[m - 1, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c:
            fb[m - 1, c:r] = (r - np.arange(c, r)) / (r - c)
    return fb


def log_mel(path, n_mels=80, n_fft=512, hop=128):
    """Log-mel spectrogram of a clip, normalised to 0..1 (rows = mel bands low->high,
    cols = time). A visualisation of the model's input, computed with numpy only."""
    a, sr = _read_wav(path)
    if len(a) < n_fft:
        a = np.pad(a, (0, n_fft - len(a)))
    win = np.hanning(n_fft).astype(np.float32)
    nframes = 1 + (len(a) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(nframes)[:, None]
    frames = a[idx] * win
    mag = np.abs(np.fft.rfft(frames, axis=1)).T  # [freq, time]
    mel = _mel_fb(n_mels, n_fft, sr) @ mag       # [mel, time]
    lm = np.log(mel + 1e-6)
    lo, hi = np.percentile(lm, 3.0), np.percentile(lm, 99.5)
    lm = np.clip((lm - lo) / max(1e-6, hi - lo), 0.0, 1.0)
    return lm ** 0.85  # mild gamma for punchier highs


def _spec_lut():
    pts = [(0.00, (13, 17, 23)), (0.18, (17, 44, 58)), (0.42, (28, 104, 128)),
           (0.68, (62, 200, 224)), (0.86, (150, 226, 236)), (1.00, (240, 205, 120))]
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        v = i / 255.0
        for j in range(len(pts) - 1):
            v0, c0 = pts[j]
            v1, c1 = pts[j + 1]
            if v0 <= v <= v1:
                f = (v - v0) / (v1 - v0)
                lut[i] = [int(c0[k] + (c1[k] - c0[k]) * f) for k in range(3)]
                break
    return lut


_LUT = _spec_lut()
_SPEC_CACHE = {}


def spec_image(clip_dir, clip, w, h):
    """Coloured, rect-sized spectrogram image for a clip (cached)."""
    key = (clip["name"], w, h)
    if key not in _SPEC_CACHE:
        lm = log_mel(clip_dir / clip["file"])
        rgb = _LUT[(lm * 255).astype(np.uint8)]     # [mel, time, 3]
        rgb = np.flipud(rgb)                          # low freq at the bottom
        img = Image.fromarray(rgb, "RGB").resize((w, h), Image.LANCZOS)
        _SPEC_CACHE[key] = img
    return _SPEC_CACHE[key]


# ------------------------------------------------------------------------- shared chrome
def header(d, W, sub):
    a = "ced.cpp"
    d.text((40, 22), a, fill=TEAL, font=font(26))
    x = 40 + d.textlength(a, font=font(26))
    d.text((x + 12, 28), "hears it", fill=INK, font=font(20))
    d.text((W - 40 - d.textlength(sub, font=font(15, False)), 30), sub, fill=DIM,
           font=font(15, False))
    d.line([40, 64, W - 40, 64], fill=GRID, width=1)


def brandline(d, W, H):
    fb = font(13, False)
    y = H - 30
    d.text((40, y), "github.com/mudler/ced.cpp", font=fb, fill=DIMMER)
    t = "Brought to you by the LocalAI team  ·  localai.io"
    d.text((W - 40 - d.textlength(t, font=fb), y), t, font=fb, fill=DIMMER)


def draw_spectrogram(cv, d, rect, clip, clip_dir, reveal):
    """Spectrogram panel: title, the coloured mel image wiped in to `reveal`, an
    unrevealed grid remainder, and a bright playhead at the wipe edge."""
    x, y, w, h = rect
    rounded(d, [x, y, x + w, y + h], 12, fill=PANEL, outline=GRID, width=1)
    pad = 16
    ix, iw = x + pad, w - 2 * pad
    d.text((ix, y + 12), "input audio", fill=DIM, font=font(13, False))
    cap = f"{clip['file'].split('/')[-1]}  ·  {clip['seconds']:.1f}s  ·  16 kHz mono"
    d.text((x + w - pad - d.textlength(cap, font=fmono(13)), y + 12), cap,
           fill=DIMMER, font=fmono(13))
    sx, sy = ix, y + 38
    sw, sh = iw, h - 84
    img = spec_image(clip_dir, clip, sw, sh)
    cut = max(1, int(sw * reveal))
    # unrevealed remainder as a faint grid bed
    d.rectangle([sx, sy, sx + sw, sy + sh], fill=(16, 21, 28))
    cv.paste(img.crop((0, 0, cut, sh)), (sx, sy))
    # playhead
    px = sx + cut
    if reveal < 1.0:
        d.line([px, sy, px, sy + sh], fill=(225, 242, 246), width=2)
        d.line([px, sy, px, sy + sh], fill=(120, 210, 224), width=1)
    d.rectangle([sx, sy, sx + sw, sy + sh], outline=GRID, width=1)
    return sx, sy, sw, sh


def draw_bars(d, rect, clip, fill, threads):
    """Top-5 prediction panel: each tag name + a confidence bar filling to its real
    value times `fill` (0..1 reveal), the #1 prediction highlighted in teal."""
    x, y, w, h = rect
    rounded(d, [x, y, x + w, y + h], 12, fill=PANEL, outline=GRID, width=1)
    pad = 18
    ix, iw = x + pad, w - 2 * pad
    d.text((ix, y + 14), "ced.cpp predictions", fill=TEAL, font=font(16))
    d.text((ix, y + 38), f"527 AudioSet classes  ·  top 5", fill=DIM, font=font(13, False))
    tags = clip["tags"]
    ry = y + 70
    row = (h - 88) / len(tags)
    fe = ease(fill)
    for i, (name, conf) in enumerate(tags):
        top = i == 0
        nm_font = font(20 if top else 17, top)
        # names + values fade in WITH the bars, so they snap after the reveal
        # rather than sitting visible at 0.00 during the listening phase.
        appear = min(1.0, fe * 1.5)
        nm_col = lerp(PANEL, TEAL if top else INK, appear)
        val_col = lerp(PANEL, TEAL if top else DIM, appear)
        # truncate long tag names to the panel
        nm = name
        while d.textlength(nm, font=nm_font) > iw - 70 and len(nm) > 4:
            nm = nm[:-2]
        if nm != name:
            nm = nm.rstrip(", ") + "."
        cy = ry + i * row
        d.text((ix, cy), nm, fill=nm_col, font=nm_font)
        vtxt = f"{conf * fe:.2f}"
        d.text((ix + iw - d.textlength(vtxt, font=fmono(16, True)), cy + 1), vtxt,
               fill=val_col, font=fmono(16, True))
        # bar
        bh = 12 if top else 9
        by = cy + (26 if top else 23)
        rounded(d, [ix, by, ix + iw, by + bh], 4, fill=(30, 37, 46))
        fw = max(2, int(iw * conf * fe))
        rounded(d, [ix, by, ix + fw, by + bh], 4, fill=(TEAL if top else TEAL_DIM))
    return ix, iw


def stat_line(d, x, y, clip, threads, on):
    """The 'tagged in N ms on CPU' line, with a green check once predictions land."""
    if not on:
        d.text((x, y), "listening", fill=DIM, font=font(16))
        return
    d.ellipse([x, y + 3, x + 16, y + 19], outline=GREEN, width=2)
    d.line([x + 4, y + 11, x + 8, y + 15], fill=GREEN, width=2)
    d.line([x + 8, y + 15, x + 13, y + 7], fill=GREEN, width=2)
    t1 = f"tagged in {clip['latency_ms']:.0f} ms"
    d.text((x + 26, y), t1, fill=GREEN, font=font(17))
    x2 = x + 26 + d.textlength(t1, font=font(17))
    t2 = f"   on CPU  ·  {threads} threads  ·  {clip['rtf']:.0f}x realtime"
    d.text((x2, y + 1), t2, fill=DIM, font=font(15, False))


# ------------------------------------------------------------------------------ frames
def beat_frame(W, H, spec, clip, clip_dir, t, beat_s, square):
    cv = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(cv)
    header(d, W, "527 sound classes  ·  ggml on CPU  ·  no Python")
    reveal = ease(min(1.0, t / (0.5 * beat_s)))
    fill = min(1.0, max(0.0, (t - 0.45 * beat_s) / (0.4 * beat_s)))
    landed = fill > 0.02
    if square:
        sp_rect = (40, 92, W - 80, 470)
        br_rect = (40, 92 + 470 + 22, W - 80, H - (92 + 470 + 22) - 150)
    else:
        sp_rect = (40, 92, 712, H - 92 - 70)
        br_rect = (768, 92, W - 40 - 768, H - 92 - 70)
    sx, sy, sw, sh = draw_spectrogram(cv, d, sp_rect, clip, clip_dir, reveal)
    draw_bars(d, br_rect, clip, fill, spec["threads"])
    stat_line(d, sp_rect[0] + 16, sp_rect[1] + sp_rect[3] - 26, clip, spec["threads"], landed)
    brandline(d, W, H)
    return cv


def title_frame(W, H, spec, t):
    cv = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(cv)
    cy = H // 2
    a = ease(min(1.0, t / 0.5))
    big = font(72)
    t1, t2 = "ced.cpp", " hears it"
    w1 = d.textlength(t1, font=big)
    w2 = d.textlength(t2, font=big)
    x0 = (W - (w1 + w2)) // 2
    d.text((x0, cy - 90), t1, fill=TEAL, font=big)
    d.text((x0 + w1, cy - 90), t2, fill=INK, font=big)
    sub = "it listens to a sound and names what it is"
    fs = font(26, False)
    d.text(((W - d.textlength(sub, font=fs)) // 2, cy + 6), sub, fill=DIM, font=fs)
    chip = f"{spec['num_classes']} AudioSet sound classes   ·   ggml on CPU   ·   no Python"
    fc = fmono(18)
    cw = d.textlength(chip, font=fc) + 44
    rounded(d, [(W - cw) // 2, cy + 60, (W + cw) // 2, cy + 100], 10,
            outline=TEAL_DIM, width=1)
    d.text(((W - d.textlength(chip, font=fc)) // 2, cy + 70), chip, fill=TEAL, font=fc)
    brandline(d, W, H)
    return cv


def end_card(W, H, spec, square):
    cv = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(cv)
    logo = Image.open(HERE / "assets" / "localai_logo.png").convert("RGBA")
    ls = 188 if square else 168
    logo = logo.resize((ls, ls), Image.LANCZOS)
    ly = int(H * (0.10 if square else 0.06))
    cv.paste(logo, ((W - ls) // 2, ly), logo)
    y = ly + ls + 6
    fteam = font(18, False)
    team = "from the LocalAI team  ·  localai.io"
    d.text(((W - d.textlength(team, font=fteam)) // 2, y), team, fill=DIM, font=fteam)
    y += 40
    head = "527-class audio tagging, on CPU"
    big = font(40)
    if d.textlength(head, font=big) > W - 80:
        big = font(34)
    d.text(((W - d.textlength(head, font=big)) // 2, y), head, fill=TEAL, font=big)
    y += 58
    sub = "from-scratch C++/ggml  ·  no Python, no PyTorch  ·  bit-exact with the original"
    fsub = font(19, False)
    d.text(((W - d.textlength(sub, font=fsub)) // 2, y), sub, fill=INK, font=fsub)
    y += 40
    cpu = f"every tag + {spec['clips'][0]['latency_ms']:.0f} ms latency above is real, measured on {spec['cpu']}"
    fcpu = font(15, False)
    d.text(((W - d.textlength(cpu, font=fcpu)) // 2, y), cpu, fill=DIMMER, font=fcpu)

    # recap of the three real results, to reinforce what the reel just showed
    ry = int(H * (0.46 if square else 0.52))
    fr = font(20)
    frc = fmono(20, True)
    cells = [(c["tags"][0][0], c["tags"][0][1]) for c in spec["clips"]]
    seg = []
    for nm, cf in cells:
        seg.append((nm + "  ", INK, fr))
        seg.append((f"{cf:.2f}", TEAL, frc))
    sep = ("      ", DIM, fr)
    parts = []
    for i, group in enumerate([seg[0:2], seg[2:4], seg[4:6]]):
        if i:
            parts.append(sep)
        parts.extend(group)
    total = sum(d.textlength(t, font=f) for t, _, f in parts)
    x = (W - total) // 2
    for t, c, f in parts:
        d.text((x, ry), t, fill=c, font=f)
        x += d.textlength(t, font=f)

    fl = font(18)
    gapx = 56

    def link_row(pair, ly):
        widths = [d.textlength(t, font=fl) for t in pair]
        total = sum(widths) + gapx * (len(pair) - 1)
        x = (W - total) // 2
        for t, w in zip(pair, widths):
            d.text((x, ly), t, fill=TEAL, font=fl)
            x += w + gapx

    link_row(["localai.io", "github.com/mudler/LocalAI"], int(H * (0.62 if square else 0.80)))
    link_row(["github.com/mudler/ced.cpp", "huggingface.co/mudler/ced.cpp-gguf"],
             int(H * (0.69 if square else 0.87)))
    return cv


# ------------------------------------------------------------------------------ render
def build_frames(W, H, spec, clip_dir, fps, title_s, beat_s, gap_s, card_s, square):
    frames = []
    for i in range(int(title_s * fps)):
        frames.append(title_frame(W, H, spec, i / fps))
    for clip in spec["clips"]:
        n = int((beat_s + gap_s) * fps)
        for i in range(n):
            frames.append(beat_frame(W, H, spec, clip, clip_dir, i / fps, beat_s, square))
    card = end_card(W, H, spec, square)
    for i in range(int(card_s * fps)):
        frames.append(card)
    return frames


def encode(out, frames, fps, gif_fps, gif_w):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for k, fr in enumerate(frames):
            fr.save(tmp / f"f{k:05d}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                        "-i", str(tmp / "f%05d.png"), "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(out)], check=True)
        pal = tmp / "pal.png"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out), "-vf",
                        f"fps={gif_fps},scale={gif_w}:-1:flags=lanczos,palettegen=stats_mode=diff",
                        str(pal)], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out), "-i", str(pal),
                        "-lavfi",
                        f"fps={gif_fps},scale={gif_w}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
                        str(out.with_suffix(".gif"))], check=True)
    print("wrote", out, "+", out.with_suffix(".gif").name)


def main():
    ap = argparse.ArgumentParser(description="ced.cpp capability reel")
    ap.add_argument("--spec", default=str(HERE / "spec.json"))
    ap.add_argument("--clips", default=str(HERE), help="dir that the spec file paths are relative to")
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--gif-fps", type=int, default=16)
    ap.add_argument("--title", type=float, default=1.1, help="title seconds")
    ap.add_argument("--beat", type=float, default=2.8, help="seconds per sound")
    ap.add_argument("--gap", type=float, default=0.4, help="hold after each sound")
    ap.add_argument("--card", type=float, default=2.7, help="end card seconds")
    ap.add_argument("--stills", action="store_true")
    a = ap.parse_args()

    spec = json.loads(Path(a.spec).read_text())
    clip_dir = Path(a.clips)
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)

    for name, (W, H), square, gw in [("ced_reel", (1280, 720), False, 960),
                                     ("ced_reel_square", (1080, 1080), True, 800)]:
        frames = build_frames(W, H, spec, clip_dir, a.fps, a.title, a.beat, a.gap,
                              a.card, square)
        encode(outdir / f"{name}.mp4", frames, a.fps, a.gif_fps, gw)
        if a.stills:
            mid = beat_frame(W, H, spec, spec["clips"][0], clip_dir, a.beat, a.beat, square)
            mid.save(outdir / f"{name}_mid.png")
            end_card(W, H, spec, square).save(outdir / f"{name}_endcard.png")


if __name__ == "__main__":
    main()
