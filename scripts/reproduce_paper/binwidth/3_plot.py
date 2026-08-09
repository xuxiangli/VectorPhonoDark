"""Draft figure fig:convergence_binwidth (six panels).

Reads output/binwidth/results.json (from 2_compute_rates.py) and plots the
relative error in the rate for each bin width against the 0.1 meV
reference, one panel per (model, material):

  fig_binwidth_{hadrophilic_heavy,hadrophilic_light,dark_photon_light}_{GaAs,Al2O3}.pdf

The JSON stores the total rate at sigma_0_bar = 1 cm^2 (events per kg-yr);
the panels show |R(dE) - R(ref)| / R(ref).

Usage:
    python scripts/reproduce_paper/binwidth/3_plot.py
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.titlesize": 13,
        "figure.dpi": 150,
        "lines.linewidth": 2.0,
    }
)

MATERIALS = ["GaAs", "Al2O3"]
MASSES_KEV = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
REF_BIN = "0p1meV"

BIN_CONFIGS = [
    ("1p0meV", "1.0 meV", "o", "#1f77b4"),
    ("0p5meV", "0.5 meV", "s", "#ff7f0e"),
    ("0p25meV", "0.25 meV", "^", "#2ca02c"),
]

MODEL_DISPLAY = {
    "hadrophilic_heavy": "heavy hadrophilic",
    "hadrophilic_light": "light hadrophilic",
    "dark_photon_light": "light dark photon",
}
MATERIAL_DISPLAY = {"GaAs": "GaAs", "Al2O3": r"Al$_2$O$_3$"}


def rel_err(val, ref):
    """Relative rate error against the reference-bin-width rate."""
    if any(v is None or np.isinf(v) or v <= 0 for v in [val, ref]):
        return float("nan")
    return abs(val - ref) / ref


def main():
    project_root = Path(__file__).resolve().parents[3]
    out_dir = project_root / "output" / "binwidth"
    results_file = out_dir / "results.json"

    if not results_file.exists():
        print(f"Results not found: {results_file}")
        print("Run scripts/reproduce_paper/binwidth/2_compute_rates.py first.")
        return
    data = json.loads(results_file.read_text())

    for material in MATERIALS:
        for model_key, model_title in MODEL_DISPLAY.items():
            model_data = data.get(material, {}).get(model_key, {})
            fig, ax = plt.subplots(figsize=(7, 5))

            for de_label, de_display, marker, color in BIN_CONFIGS:
                masses, errors = [], []
                for mass_keV in MASSES_KEV:
                    mk = f"{mass_keV}keV"
                    err = rel_err(
                        model_data.get(mk, {}).get(de_label),
                        model_data.get(mk, {}).get(REF_BIN),
                    )
                    if not np.isnan(err):
                        masses.append(mass_keV)
                        errors.append(err)
                ax.plot(
                    masses,
                    errors,
                    color=color,
                    marker=marker,
                    ms=6,
                    lw=1.8,
                    label=rf"$\Delta E = {de_display}$",
                )

            ax.set_xlabel(r"DM mass  $m_\chi$  [keV]")
            ax.set_ylabel("Relative error")
            ax.set_title(f"{MATERIAL_DISPLAY[material]} {model_title}")
            ax.set_xticks(MASSES_KEV)
            ax.set_yscale("log")
            ax.grid(True, which="both", linestyle=":", alpha=0.5)
            ax.legend(loc="best")
            fig.tight_layout()

            p = out_dir / f"fig_binwidth_{model_key}_{material}.pdf"
            fig.savefig(p, bbox_inches="tight")
            print(f"Saved {p}")
            plt.close(fig)


if __name__ == "__main__":
    main()
