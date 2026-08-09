"""
Figure: first eight radial basis functions of the linear vs logarithmic
spherical Haar wavelet bases (Sec. 3.3, fig:wavelet_shapes).

Stacked/offset layout: each wavelet n = 0..7 sits on its own baseline row
(y = -n), scaled to a common display height so its shape is visible without
overlap. The true normalization factors A_n, B_n are not drawn -- only the
sign of each lobe is used; their growth toward small x is described in the
caption instead.

Both panels share a linear x axis on [0, 1].

Saved to: output/wavelet/fig_wavelet_shapes.pdf

Usage:
    python scripts/reproduce_paper/wavelet/3_plot_shapes.py
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

from vectorphonodark import basis_funcs as bf

# =====================  Configuration  =====================

# wavelets n = 0 .. NUM-1
NUM = 8
# IR cutoff of the logarithmic basis (illustration)
EPS = 1e-2
# display half-height of each lobe (fraction of unit row pitch)
HEIGHT = 0.40

OUT = (
    Path(__file__).resolve().parents[3]
    / "output"
    / "wavelet"
    / "fig_wavelet_shapes.pdf"
)


# =====================  Wavelet geometry  =====================


def wavelet_shape(n, basis):
    """Return (x1, x2, x3, sign_lobe1, sign_lobe2) for wavelet n.

    x2 is None for the n = 0 top-hat (single lobe). Only the signs of the
    lobes are kept; the true amplitudes A_n, B_n are replaced by HEIGHT.
    """
    if basis == "linear":
        if n == 0:
            return 0.0, None, 1.0, +1, None
        x1, x2, x3 = bf.haar_support(n)
        _, b = bf.haar_value(n, dim=3)  # b < 0
    else:
        if n == 0:
            return EPS, None, 1.0, +1, None
        x1, x2, x3 = bf.haar_support_log(n, EPS)
        _, b = bf.haar_value_log(n, EPS, p=2)  # b < 0
    return x1, x2, x3, +1, int(np.sign(b))


def step_curve(x1, x2, x3, s1, s2, y0):
    """Trace the wavelet as a step curve offset to baseline y0."""
    if x2 is None:  # n = 0 top-hat
        xs = [x1, x1, x3, x3]
        ys = [y0, y0 + s1 * HEIGHT, y0 + s1 * HEIGHT, y0]
    else:
        xs = [x1, x1, x2, x2, x3, x3]
        ys = [
            y0,
            y0 + s1 * HEIGHT,
            y0 + s1 * HEIGHT,
            y0 + s2 * HEIGHT,
            y0 + s2 * HEIGHT,
            y0,
        ]
    return np.array(xs), np.array(ys)


# =====================  Figure  =====================

fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2), sharey=True)
colors = plt.cm.viridis(np.linspace(0.0, 0.85, NUM))

for ax, basis in zip(axes, ["linear", "log"]):
    for n in range(NUM):
        y0 = -float(n)
        ax.axhline(y0, color="0.88", lw=0.6, zorder=0)
        x1, x2, x3, s1, s2 = wavelet_shape(n, basis)
        xs, ys = step_curve(x1, x2, x3, s1, s2, y0)
        ax.plot(xs, ys, color=colors[n], lw=1.5, zorder=3)
        ax.fill_between(xs, y0, ys, color=colors[n], alpha=0.35, zorder=2)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(r"$x$")

axes[0].set_yticks([-n for n in range(NUM)])
axes[0].set_yticklabels([rf"$n={n}$" for n in range(NUM)])
axes[0].set_ylim(-(NUM - 1) - HEIGHT - 0.2, HEIGHT + 0.2)
for ax in axes:
    ax.tick_params(axis="y", length=0)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)

axes[0].set_title("linear wavelets")
axes[1].set_title(r"logarithmic wavelets  ($\epsilon = 10^{-2}$)")

# ---  Generation column: label lambda to the left, bracket multi-wavelet ones  ---
#   n = 0 is the constant (mean); wavelets n >= 1 have generation lambda = floor(log2 n),
#   which contains 2^lambda wavelets.
GENERATIONS = [
    ("", [0]),
    (r"$\lambda=0$", [1]),
    (r"$\lambda=1$", [2, 3]),
    (r"$\lambda=2$", [4, 5, 6, 7]),
]
# spine of the bracket, in axes-fraction of the left panel
BRACKET_X = -0.15
# cap ends (point toward the n labels)
BRACKET_CAP = -0.125
# generation label, left of the bracket
LABEL_X = -0.175

trans = blended_transform_factory(axes[0].transAxes, axes[0].transData)
for label, ns in GENERATIONS:
    y_top = -min(ns) + HEIGHT + 0.05
    y_bot = -max(ns) - HEIGHT - 0.05
    y_mid = 0.5 * (y_top + y_bot)
    if len(ns) > 1:
        axes[0].plot(
            [BRACKET_X, BRACKET_X],
            [y_top, y_bot],
            color="0.4",
            lw=1.0,
            transform=trans,
            clip_on=False,
        )
        for y_end in (y_top, y_bot):
            axes[0].plot(
                [BRACKET_X, BRACKET_CAP],
                [y_end, y_end],
                color="0.4",
                lw=1.0,
                transform=trans,
                clip_on=False,
            )
    axes[0].text(
        LABEL_X, y_mid, label, transform=trans, ha="right", va="center", fontsize=10
    )

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print("saved", OUT)
