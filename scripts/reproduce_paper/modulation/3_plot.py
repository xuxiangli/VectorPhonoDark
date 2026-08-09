"""
Al2O3 daily-modulation benchmark (light dark photon, omega_min = 20 meV):
figure and comparison table.

Reads the energy-binned rates from 1b_vpd_rate.py (VectorPhonoDark) and
2_pd.py (PhonoDark), imposes the 20 meV threshold by summing only the stored
bins at or above it -- no new projection or PhonoDark run -- converts both to
the rate R(t) at a common reference cross section, and forms the
daily-modulation ratio R(t)/<R>.

Reproduces the draft figure (fig:modulation_al2o3_20meV) and its table
(tab:modulation_20meV): the VPD/PD ratios of the R/<R> minimum, of the R/<R>
maximum, and of the daily average <R>. The full per-hour comparison data is
written alongside as JSON.

Pipeline:   1a_vpd_project.py  ->  1b_vpd_rate.py  ->  2_pd.py  ->  3_plot.py
Inputs:     output/modulation/{vpd_binned.json, pd_binned.json}
Outputs:    output/modulation/fig_modulation_al2o3_20meV.pdf
            output/modulation/modulation_comparison_20meV.json

Run (vectorphonodark environment):
    python -u scripts/reproduce_paper/modulation/3_plot.py
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from vectorphonodark import constants as const

THRESHOLD_MILLIEV = 20.0
MASSES_KEV = [50, 100, 500, 1000]
COLORS = ["C0", "C1", "C2", "C3"]


def mass_label(m_keV: int) -> str:
    return f"{m_keV} keV" if m_keV < 1000 else "1 MeV"


def sum_above_threshold(binned, lower_edges_meV, threshold_meV):
    """Sum the energy bins whose lower edge is at or above the threshold."""
    binned = np.asarray(binned, dtype=float)  # shape: (n_bins, n_times)
    edges = np.asarray(lower_edges_meV, dtype=float)  # shape: (n_bins,)
    return binned[edges >= threshold_meV - 1e-9].sum(axis=0)  # shape: (n_times,)


def vpd_rate(total_gamma):
    """VPD rate R(t) [events/kg/yr at sigma_0_bar = 1 cm^2] from the summed
    rate response Gamma(t)."""
    return const.KG_YR * const.RHO_DM * total_gamma / const.INVEV_TO_CM**2


def pd_rate(total_rate, mass_eV, q0_fdm):
    """PhonoDark rate R(t) [events/kg/yr at sigma_0_bar = 1 cm^2] for the
    light mediator."""
    mu = const.M_ELEC * mass_eV / (const.M_ELEC + mass_eV)
    return const.KG_YR * np.pi * q0_fdm**4 * total_rate / mu**2 / const.INVEV_TO_CM**2


def main() -> None:
    out_dir = Path(__file__).resolve().parents[3] / "output" / "modulation"
    vpd = json.loads((out_dir / "vpd_binned.json").read_text())
    pd = json.loads((out_dir / "pd_binned.json").read_text())

    vpd_edges = vpd["bin_lower_edges_meV"]
    pd_edges = pd["bin_lower_edges_meV"]

    # R(t) at the 20 meV threshold for each code.
    R_vpd, R_pd = {}, {}
    for m in MASSES_KEV:
        g_vpd = sum_above_threshold(
            vpd["rate_response"][str(m)], vpd_edges, THRESHOLD_MILLIEV
        )
        rate_pd = sum_above_threshold(
            pd["binned_rate"][str(m)], pd_edges, THRESHOLD_MILLIEV
        )
        R_vpd[m] = vpd_rate(g_vpd)
        R_pd[m] = pd_rate(rate_pd, m * 1e3, pd["q0_fdm_eV"])

    # ---- Comparison table: draft tab:modulation_20meV + full per-hour data ----
    table = {}
    print(f"{'-' * 66}")
    print(
        f"  Al2O3 daily modulation, omega_min = {THRESHOLD_MILLIEV:.0f} meV : VPD / PD"
    )
    print(f"  {'Mass':>7s} | {'<R>':>10s} {'min':>10s} {'max':>10s}")
    print(f"{'-' * 66}")
    for m in MASSES_KEV:
        r_vpd = R_vpd[m] / R_vpd[m].mean()
        r_pd = R_pd[m] / R_pd[m].mean()
        ratio_avg = float(R_vpd[m].mean() / R_pd[m].mean())
        ratio_min = float(r_vpd.min() / r_pd.min())
        ratio_max = float(r_vpd.max() / r_pd.max())
        table[str(m)] = {
            "r_over_mean_vpd": r_vpd.tolist(),  # each hour
            "r_over_mean_pd": r_pd.tolist(),  # each hour
            "min_vpd": float(r_vpd.min()),
            "max_vpd": float(r_vpd.max()),
            "min_pd": float(r_pd.min()),
            "max_pd": float(r_pd.max()),
            "ratio_avg_R": ratio_avg,  # <R> VPD / PD
            "ratio_min": ratio_min,  # [R/<R>]_min VPD / PD
            "ratio_max": ratio_max,  # [R/<R>]_max VPD / PD
        }
        print(
            f"  {mass_label(m):>7s} | {ratio_avg:>10.4f} {ratio_min:>10.4f} "
            f"{ratio_max:>10.4f}"
        )
    print(f"{'-' * 66}")

    json_out = out_dir / "modulation_comparison_20meV.json"
    json_out.write_text(
        json.dumps(
            {
                "threshold_meV": THRESHOLD_MILLIEV,
                "masses_keV": MASSES_KEV,
                "columns": "ratios are VectorPhonoDark / PhonoDark",
                "table": table,
            },
            indent=2,
        )
    )
    print(f"Comparison data -> {json_out}")

    # ---- Figure: R(t)/<R>, PhonoDark dashed, VectorPhonoDark solid ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    times = np.arange(vpd["n_times"])
    for m, color in zip(MASSES_KEV, COLORS):
        ax.plot(times, R_pd[m] / R_pd[m].mean(), color=color, lw=1.8, ls="--")
        ax.plot(times, R_vpd[m] / R_vpd[m].mean(), color=color, lw=1.8, ls="-")
    ax.axhline(1, color="k", lw=0.7, ls=":")
    ax.set_xlabel("Time [h]", fontsize=12)
    ax.set_ylabel(r"$R(t)\,/\,\langle R \rangle$", fontsize=12)
    ax.set_title(r"Al$_2$O$_3$ light dark photon daily modulation", fontsize=11)
    ax.set_xlim(0, vpd["n_times"] - 1)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    code_lines = [
        Line2D([0], [0], color="k", lw=1.8, ls="--", label="PhonoDark"),
        Line2D([0], [0], color="k", lw=1.8, ls="-", label="VectorPhonoDark"),
    ]
    mass_lines = [
        Line2D([0], [0], color=COLORS[i], lw=2, label=mass_label(m))
        for i, m in enumerate(MASSES_KEV)
    ]
    ax.legend(handles=code_lines + mass_lines, fontsize=9, ncol=2, loc="best")
    fig.tight_layout()
    pdf_out = out_dir / "fig_modulation_al2o3_20meV.pdf"
    fig.savefig(pdf_out, dpi=150)
    plt.close(fig)
    print(f"Figure -> {pdf_out}")


if __name__ == "__main__":
    main()
