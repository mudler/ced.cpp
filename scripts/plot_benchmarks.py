#!/usr/bin/env python3
"""Render the README benchmark plots from the measured numbers in docs/BENCHMARKS.md.

Two figures, branded to match the demo:
  media/bench.png  - latency + peak memory, ced.cpp f16/q8_0 vs PyTorch CED (CPU)
  media/family.png - the CED family GGUF sizes (q8_0 / f16) across the four sizes

Numbers: ced-base, 10 s clip, AMD Ryzen 9 9950X3D, 4 threads (see docs/BENCHMARKS.md).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

BG = "#0d1117"
INK = "#d7dde5"
DIM = "#8b949e"
CYAN = "#3ec8e0"      # ced.cpp
CYAN2 = "#7ee0ef"     # ced.cpp (lighter, q8)
GRAY = "#6e7681"      # PyTorch
GRID = "#21262d"

# Prefer a mono font to echo the CLI/demo look; fall back to the default.
for cand in ("DejaVu Sans Mono", "DejaVu Sans"):
    try:
        plt.rcParams["font.family"] = font_manager.FontProperties(family=cand).get_name()
        break
    except Exception:
        pass
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": DIM,
    "axes.edgecolor": GRID, "axes.linewidth": 0.0,
})


def style(ax, title):
    ax.set_title(title, color=INK, fontsize=12, pad=12, loc="left", fontweight="bold")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    ax.set_yticklabels([])


def bars(ax, labels, values, colors, fmt, sub=None):
    x = range(len(labels))
    b = ax.bar(x, values, color=colors, width=0.62, zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10.5)
    top = max(values)
    for i, (rect, v) in enumerate(zip(b, values)):
        ax.text(rect.get_x() + rect.get_width() / 2, v + top * 0.02, fmt(v),
                ha="center", va="bottom", color=INK, fontsize=11, fontweight="bold")
        if sub:
            ax.text(rect.get_x() + rect.get_width() / 2, v * 0.5, sub[i],
                    ha="center", va="center", color=BG, fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, top * 1.18)


# ---- figure 1: speed + memory --------------------------------------------
# Lead with the apples-to-apples PyTorch f32 vs ced.cpp f32 pair; f16/q8_0 are
# the near-lossless quantized configs you'd actually ship.
CYAN0 = "#2596be"    # ced.cpp f32 (darker cyan, the same-precision comparison)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
labels = ["PyTorch\nf32", "ced.cpp\nf32", "ced.cpp\nf16", "ced.cpp\nq8_0"]
cols = [GRAY, CYAN0, CYAN, CYAN2]

style(a1, "Latency per 10 s clip  (lower is better)")
bars(a1, labels, [158.8, 126.6, 102.9, 117.1], cols, lambda v: f"{v:.0f} ms",
     sub=["64x RT", "80x RT", "98x RT", "86x RT"])

style(a2, "Peak memory  (lower is better)")
bars(a2, labels, [717, 354, 189, 111], cols, lambda v: f"{v:.0f} MB")

fig.suptitle("ced.cpp vs PyTorch CED on CPU  -  ced-base, Ryzen 9 9950X3D, 4 threads",
             color=DIM, fontsize=11, y=1.0, x=0.5)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig("media/bench.png", dpi=150)
print("wrote media/bench.png")

# ---- figure 2: family sizes ----------------------------------------------
fig2, ax = plt.subplots(figsize=(8.8, 3.8))
fam = ["ced-tiny\n5.5M", "ced-mini\n9.6M", "ced-small\n22M", "ced-base\n86M"]
q8 = [6, 11, 23, 88]
f16 = [11, 19, 42, 165]
x = range(len(fam))
w = 0.38
ax.bar([i - w / 2 for i in x], f16, width=w, color=CYAN, zorder=3, label="f16")
ax.bar([i + w / 2 for i in x], q8, width=w, color=CYAN2, zorder=3, label="q8_0")
style(ax, "CED family GGUF size  (MB)")
ax.set_xticks(list(x)); ax.set_xticklabels(fam, fontsize=10)
for i in x:
    ax.text(i - w / 2, f16[i] + 4, f"{f16[i]}", ha="center", color=INK, fontsize=9.5)
    ax.text(i + w / 2, q8[i] + 4, f"{q8[i]}", ha="center", color=INK, fontsize=9.5)
ax.set_ylim(0, max(f16) * 1.16)
leg = ax.legend(loc="upper left", frameon=False, fontsize=10, labelcolor=INK)
fig2.tight_layout()
fig2.savefig("media/family.png", dpi=150)
print("wrote media/family.png")
