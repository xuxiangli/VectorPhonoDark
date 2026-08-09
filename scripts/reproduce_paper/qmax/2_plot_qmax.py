"""Draft figures fig:qmax_heavy and fig:qmax_light.

Reads output/qmax/results.csv (from 1_run_qmax.py) and plots the
relative error of the rate computed from the universal q_max = q_cut
projection against the per-mass reference, as a function of
k(m_chi) = q_max(m_chi) / q_cut:

  fig_qmax_heavy.pdf  heavy hadrophilic, linear (dashed) vs log (solid)
  fig_qmax_light.pdf  light hadrophilic and dark photon, log basis only
                      (the linear basis has no converged result to show)

The shaded band marks the 1e-3 accuracy of the reference itself.

Usage:
    python scripts/reproduce_paper/qmax/2_plot_qmax.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =====================  Style  =====================

# Color encodes the material (blue GaAs, red Al2O3); the shade encodes the
# coupling (darker = hadrophilic, lighter = dark photon).
COLOR_MATERIAL = {
    "GaAs": "#1f77b4",  # blue
    "Al2O3": "#d62728",  # red
}
COLOR_LIGHT = {
    ("light_hadrophilic", "GaAs"): "#1f77b4",  # blue
    ("light_dark_photon", "GaAs"): "#aec7e8",  # light blue
    ("light_hadrophilic", "Al2O3"): "#d62728",  # red
    ("light_dark_photon", "Al2O3"): "#ff9896",  # light red
}
LS_WAVELET = {"linear": "--", "log": "-"}
MARKER_WAVELET = {"linear": "s", "log": "o"}
MARKER_MODEL = {"light_hadrophilic": "o", "light_dark_photon": "^"}
MARKERSIZE = 4

MATERIAL_DISPLAY = {"GaAs": "GaAs", "Al2O3": "Al$_2$O$_3$"}
MODEL_DISPLAY = {"light_hadrophilic": "hadrophilic", "light_dark_photon": "dark photon"}


# =====================  Data  =====================


def load_curves(csv_path):
    """(material, model) -> sorted [(k, rel_err), ...], zero errors dropped."""
    curves = {}
    with open(str(csv_path), newline="") as f:
        for row in csv.DictReader(f):
            rate_ref = float(row["rate_ref"])
            if rate_ref == 0.0:
                continue
            err = abs(float(row["rate_universal"]) - rate_ref) / abs(rate_ref)
            if err == 0.0:
                continue  # undefined on a log scale
            k = float(row["q_max_ref"]) / float(row["q_cut"])
            curves.setdefault((row["material"], row["model"]), []).append((k, err))
    return {key: sorted(pts) for key, pts in curves.items()}


def get_curve(curves, material, model):
    pts = curves.get((material, model), [])
    return [p[0] for p in pts], [p[1] for p in pts]


# =====================  Figure helpers  =====================


def _finish_axes(ax, ylim):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$k(m_\chi)$", fontsize=13)
    ax.set_ylabel("Relative error", fontsize=13)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.set_xlim(right=1.0)
    ax.axhspan(ylim[0], 1e-3, alpha=0.08, color="gray", zorder=0)
    ax.axhline(1e-3, color="gray", ls="--", lw=1.0, zorder=1)
    ax.set_ylim(*ylim)


def make_figure_heavy(curves, out_path):
    """Heavy hadrophilic: 2 materials x 2 bases at q_max = q_cut."""
    fig, ax = plt.subplots(figsize=(7, 5))

    models = {"linear": "heavy_hadrophilic_linear", "log": "heavy_hadrophilic_log"}
    for material in ("GaAs", "Al2O3"):
        for wavelet, model in models.items():
            xs, ys = get_curve(curves, material, model)
            if not xs:
                continue
            ax.plot(
                xs,
                ys,
                color=COLOR_MATERIAL[material],
                linestyle=LS_WAVELET[wavelet],
                marker=MARKER_WAVELET[wavelet],
                markersize=MARKERSIZE,
                linewidth=1.5,
            )

    ax.set_title("One projection for all masses: heavy mediator", fontsize=11)
    _finish_axes(ax, (1e-5, 1e2))

    mat_handles = [
        Line2D(
            [0],
            [0],
            color=COLOR_MATERIAL[m],
            lw=2,
            label=f"{MATERIAL_DISPLAY[m]}, hadrophilic",
        )
        for m in ("GaAs", "Al2O3")
    ]
    basis_handles = [
        Line2D(
            [0],
            [0],
            color="k",
            lw=1.5,
            ls=LS_WAVELET[w],
            marker=MARKER_WAVELET[w],
            markersize=MARKERSIZE,
            label=f"{w} wavelet",
        )
        for w in ("linear", "log")
    ]
    leg1 = ax.legend(
        handles=mat_handles,
        loc="upper right",
        fontsize=9,
        title="material, model",
        title_fontsize=9,
        framealpha=0.85,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=basis_handles,
        loc="lower left",
        fontsize=9,
        title="basis",
        title_fontsize=9,
        framealpha=0.85,
    )

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def make_figure_light(curves, out_path):
    """Light mediators: 2 models x 2 materials, log basis only."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for model in ("light_hadrophilic", "light_dark_photon"):
        for material in ("GaAs", "Al2O3"):
            xs, ys = get_curve(curves, material, model)
            if not xs:
                continue
            ax.plot(
                xs,
                ys,
                color=COLOR_LIGHT[(model, material)],
                linestyle=LS_WAVELET["log"],
                marker=MARKER_MODEL[model],
                markersize=MARKERSIZE,
                linewidth=1.5,
            )

    ax.set_title("One projection for all masses: light mediator", fontsize=11)
    _finish_axes(ax, (1e-5, 1e2))

    handles = [
        Line2D(
            [0],
            [0],
            color=COLOR_LIGHT[(model, material)],
            lw=2,
            marker=MARKER_MODEL[model],
            markersize=MARKERSIZE,
            label=f"{MATERIAL_DISPLAY[material]}, {MODEL_DISPLAY[model]}",
        )
        for model in ("light_hadrophilic", "light_dark_photon")
        for material in ("GaAs", "Al2O3")
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        fontsize=8,
        title="material, model",
        title_fontsize=8,
        framealpha=0.85,
    )

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =====================  Main  =====================


def main():
    project_root = Path(__file__).resolve().parents[3]
    cmp_dir = project_root / "output" / "qmax"
    csv_path = cmp_dir / "results.csv"

    if not csv_path.exists():
        print(f"Results not found: {csv_path}")
        print("Run scripts/reproduce_paper/qmax/1_run_qmax.py first.")
        return

    curves = load_curves(csv_path)
    print(f"Loaded {sum(len(v) for v in curves.values())} points from {csv_path}")

    make_figure_heavy(curves, cmp_dir / "fig_qmax_heavy.pdf")
    make_figure_light(curves, cmp_dir / "fig_qmax_light.pdf")


if __name__ == "__main__":
    main()
