"""Draft figures fig:wavelet_comparison_{light,heavy}.

Reads output/wavelet/results.csv (from 1_run_wavelet.py) and plots the
relative error of the rate against N_q = nq_max + 1, for four DM masses and
both wavelet bases. Each basis uses its own N_q = 2048 result as reference.

  fig_wavelet_light.pdf  light mediator (GaAs, hadrophilic)
  fig_wavelet_heavy.pdf  heavy mediator

Usage:
    python scripts/reproduce_paper/wavelet/2_plot_wavelet.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =====================  Configuration  =====================

MASSES_MEV = [0.1, 1.0, 10.0, 100.0]
# N_q = 1 .. 2048
NQ_MAX_LIST = [2**k - 1 for k in range(12)]
NQ_MAX_REF = 2047

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # blue, orange, green, red
LINESTYLE = {"log": "-", "linear": "--"}
MARKER = {"log": "o", "linear": "s"}
MARKERSIZE = 5


# =====================  Data  =====================


def load_results(csv_path):
    """(wavelet, mediator, mass_MeV, nq_max) -> rate."""
    results = {}
    with open(str(csv_path), newline="") as f:
        for row in csv.DictReader(f):
            key = (
                row["wavelet"],
                row["mediator"],
                float(row["mass_MeV"]),
                int(row["nq_max"]),
            )
            results[key] = float(row["rate"])
    return results


# =====================  Figures  =====================


def make_figure(results, mediator, title, out_path):
    """Relative error vs N_q for all masses and both bases."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for i_mass, mass_mev in enumerate(MASSES_MEV):
        for wavelet in ("log", "linear"):
            rate_ref = results.get((wavelet, mediator, mass_mev, NQ_MAX_REF))
            if rate_ref is None or rate_ref == 0.0:
                print(
                    f"  WARNING: reference rate missing for "
                    f"{wavelet}/{mediator}/{mass_mev} MeV  -- skipping"
                )
                continue

            x_vals, y_vals = [], []
            for nq_max in NQ_MAX_LIST:
                if nq_max == NQ_MAX_REF:
                    continue
                rate = results.get((wavelet, mediator, mass_mev, nq_max))
                if rate is None:
                    continue
                rel_err = abs(rate - rate_ref) / abs(rate_ref)
                if rel_err > 0.0:
                    x_vals.append(nq_max + 1)  # N_q = nq_max + 1
                    y_vals.append(rel_err)

            if x_vals:
                ax.plot(
                    x_vals,
                    y_vals,
                    linestyle=LINESTYLE[wavelet],
                    marker=MARKER[wavelet],
                    markersize=MARKERSIZE,
                    color=COLORS[i_mass],
                    linewidth=1.5,
                )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$N_q$", fontsize=13)
    ax.set_ylabel("Relative error", fontsize=13)
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    mass_handles = [
        Line2D([0], [0], color=COLORS[i], lw=2, label=f"{m:.1f} MeV")
        for i, m in enumerate(MASSES_MEV)
    ]
    style_handles = [
        Line2D(
            [0],
            [0],
            color="k",
            lw=1.5,
            ls=LINESTYLE[w],
            marker=MARKER[w],
            markersize=MARKERSIZE,
            label=f"{w} wavelet",
        )
        for w in ("log", "linear")
    ]
    leg1 = ax.legend(
        handles=mass_handles,
        loc="upper right",
        fontsize=9,
        title="DM mass",
        title_fontsize=9,
        framealpha=0.8,
    )
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc="lower left", fontsize=9, framealpha=0.8)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =====================  Main  =====================


def main():
    project_root = Path(__file__).resolve().parents[3]
    wc_dir = project_root / "output" / "wavelet"
    csv_path = wc_dir / "results.csv"

    if not csv_path.exists():
        print(f"Results not found: {csv_path}")
        print("Run scripts/reproduce_paper/wavelet/1_run_wavelet.py first.")
        return

    results = load_results(csv_path)
    print(f"Loaded {len(results)} data points from {csv_path}")

    make_figure(
        results,
        mediator="light",
        title="Convergence: GaAs light hadrophilic",
        out_path=wc_dir / "fig_wavelet_light.pdf",
    )
    make_figure(
        results,
        mediator="heavy",
        title="Convergence: GaAs heavy hadrophilic",
        out_path=wc_dir / "fig_wavelet_heavy.pdf",
    )


if __name__ == "__main__":
    main()
