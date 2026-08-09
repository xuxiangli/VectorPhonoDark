"""Draft appendix tables tab:conv2_* from the convergence study.

Reads output/convergence/results.csv (from 1_run_convergence.py), computes
the relative error of each scanned value against its scan's reference, and
writes the seven LaTeX tables to output/convergence/convergence_tables.tex.
The same numbers are printed as text for a quick look.

Usage:
    python scripts/reproduce_paper/convergence/2_make_tables.py [results.csv]
"""

import csv
import sys
from pathlib import Path

MATERIALS = ["GaAs", "Al2O3"]
MODELS = ["heavy_hadrophilic", "light_hadrophilic", "light_dark_photon"]
MASSES_MEV = [0.1, 1.0, 10.0, 100.0]

MATERIAL_TEX = {"GaAs": "GaAs", "Al2O3": "Al$_2$O$_3$"}
MODEL_TEX = {
    "heavy_hadrophilic": "heavy, hadr.",
    "light_hadrophilic": "light, hadr.",
    "light_dark_photon": "light, DP",
}

# One spec per table: which scan, which rows (in scan values), the reference
# value, the column-header symbol, how a value is displayed (N = n + 1 for
# wavelet counts), the float placement, and the caption. Wording and styling
# mirror the draft appendix (app:convergence) so regenerated tables diff
# cleanly against it.
TABLES = [
    {
        "scan": "l_max",
        "label": "tab:conv2_lmax",
        "header": r"$\ell_\text{max}$",
        "rows": [2, 5],
        "ref": 8,
        "display": lambda v: f"{v:g}",
        "placement": "tbp",
        "caption": (
            r"Relative error in the rate vs.\ the reference ($\ell_\text{max}=8$)"
            "\n    "
            r"as a function of $\ell_\text{max}$, for all material/model "
            "combinations\n    "
            r"and four DM masses. ``hadr.''\ = hadrophilic; ``DP'' = dark photon."
            "\n    All other parameters are held at their reference values."
        ),
    },
    {
        "scan": "n_vmax",
        "label": "tab:conv2_nvmax",
        "header": r"$N_v$",
        "rows": [63, 127],
        "ref": 255,
        "display": lambda v: f"{v + 1:g}",
        "placement": "htbp",
        "caption": (
            r"Relative error in the rate vs.\ the reference "
            r"($N_v=256$ with $N_r^\text{VDF}=256$)"
            "\n    "
            r"as a function of $N_v$, for all material/model combinations"
            "\n    and four DM masses."
            "\n    All other parameters are held at their reference values."
        ),
    },
    {
        "scan": "n_qmax",
        "label": "tab:conv2_nqmax",
        "header": r"$N_q$",
        "rows": [511, 1023],
        "ref": 2047,
        "display": lambda v: f"{v + 1:g}",
        "placement": "htbp",
        "caption": (
            r"Relative error in the rate vs.\ the reference "
            r"($N_q=2048$ with $N_r^\text{FF}=2048$)"
            "\n    "
            r"as a function of the number of radial wavelets $N_q$ used to"
            "\n    expand the material form factor,"
            "\n    for all material/model combinations and four DM masses."
            "\n    All other parameters are held at their reference values."
        ),
    },
    {
        "scan": "vdf_n_r",
        "label": "tab:conv2_vdf_nr",
        "header": r"$N_r^\text{VDF}$",
        "rows": [128, 256],
        "ref": 512,
        "display": lambda v: f"{v:g}",
        "placement": "htbp",
        "caption": (
            r"Relative error in the rate vs.\ the reference ($N_r^\text{VDF}=512$)"
            "\n    as a function of the VDF radial grid size,"
            "\n    for all material/model combinations and four DM masses."
            "\n    All other parameters are held at their reference values."
        ),
    },
    {
        "scan": "vdf_n_ang",
        "label": "tab:conv2_vdf_nang",
        "header": r"$N_\Omega^\text{VDF}$",
        "rows": [90, 180],
        "ref": 360,
        "display": lambda v: f"{v:g}",
        "placement": "htbp",
        "caption": (
            r"Relative error in the rate vs.\ the reference "
            r"($N_\Omega^\text{VDF}=360$)"
            "\n    as a function of the VDF angular grid size,"
            "\n    for all material/model combinations and four DM masses."
            "\n    All other parameters are held at their reference values."
        ),
    },
    {
        "scan": "ff_n_r",
        "label": "tab:conv2_ff_nr_finest",
        "header": r"$N_r^\text{FF}$",
        "rows": [512, 1024],
        "ref": 2048,
        "display": lambda v: f"{v:g}",
        "placement": "tbp",
        "caption": (
            r"Relative error in the rate vs.\ the reference ($N_r^\text{FF}=2048$)"
            "\n    as a function of the material form factor radial grid size,"
            "\n    for all material/model combinations and four DM masses."
            "\n    All other parameters are held at their reference values."
        ),
    },
    {
        "scan": "ff_n_ang",
        "label": "tab:conv2_ff_nang",
        "header": r"$N_\Omega^\text{FF}$",
        "rows": [15, 25],
        "ref": 50,
        "display": lambda v: f"{v:g}",
        "placement": "htbp",
        "caption": (
            r"Relative error in the rate vs.\ the reference "
            r"($N_\Omega^\text{FF}=50$)"
            "\n    for the material form factor projection,"
            "\n    as a function of the material form factor angular grid size,"
            "\n    for all material/model combinations and four DM masses."
            "\n    All other parameters are held at their reference values."
        ),
    },
]


def load_rates(csv_path):
    """(scan, material, model, mass_MeV, value) -> rate."""
    rates = {}
    with open(str(csv_path), newline="") as f:
        for row in csv.DictReader(f):
            key = (
                row["scan"],
                row["material"],
                row["model"],
                float(row["mass_MeV"]),
                float(row["value"]),
            )
            rates[key] = float(row["rate"])
    return rates


def rel_err_pct(rates, spec, material, model, mass, value):
    rate = rates.get((spec["scan"], material, model, mass, float(value)))
    ref = rates.get((spec["scan"], material, model, mass, float(spec["ref"])))
    if rate is None or ref is None or ref == 0.0:
        return None
    return abs(rate - ref) / abs(ref) * 100.0


def make_table(rates, spec):
    """One LaTeX table in the draft's format; also returns the cell values."""
    n_rows = len(spec["rows"])
    lines = [
        f"\\begin{{table}}[{spec['placement']}]",
        r"  \centering",
        r"  \renewcommand{\arraystretch}{1.2}",
        r"  \begin{tabular}{llccccc}",
        r"    \hline\hline",
        f"    Material & Model & {spec['header']} &",
        r"      $0.1$~MeV & $1$~MeV &",
        r"      $10$~MeV & $100$~MeV \\",
        r"    \hline",
    ]
    cells = {}
    for i_mat, material in enumerate(MATERIALS):
        lines.append(
            f"    \\multirow{{{n_rows * len(MODELS)}}}{{*}}{{{MATERIAL_TEX[material]}}}"
        )
        for i_mod, model in enumerate(MODELS):
            lines.append(f"      & \\multirow{{{n_rows}}}{{*}}{{{MODEL_TEX[model]}}}")
            for i_val, value in enumerate(spec["rows"]):
                entries = []
                for mass in MASSES_MEV:
                    err = rel_err_pct(rates, spec, material, model, mass, value)
                    cells[(material, model, value, mass)] = err
                    entries.append("--" if err is None else f"{err:.2f}\\%")
                prefix = "           " if i_val == 0 else "      &    "
                lines.append(
                    f"{prefix}& {spec['display'](value)} & "
                    + " & ".join(entries)
                    + r" \\"
                )
            if i_mod < len(MODELS) - 1:
                lines.append(r"    \cline{2-7}")
        lines.append(r"    \hline" + (r"\hline" if i_mat == 1 else ""))
    lines += [
        r"  \end{tabular}",
        f"  \\caption{{{spec['caption']}}}",
        f"  \\label{{{spec['label']}}}",
        r"\end{table}",
    ]
    return "\n".join(lines), cells


def print_summary(spec, cells):
    print(f"\n=== {spec['label']} (scan: {spec['scan']}, ref = {spec['ref']}) ===")
    header = f"  {'material':<8} {'model':<18} {'value':>6}" + "".join(
        f"  {m:g} MeV".rjust(10) for m in MASSES_MEV
    )
    print(header)
    for material in MATERIALS:
        for model in MODELS:
            for value in spec["rows"]:
                row = f"  {material:<8} {model:<18} {spec['display'](value):>6}"
                for mass in MASSES_MEV:
                    err = cells.get((material, model, value, mass))
                    row += f"  {'--':>8}" if err is None else f"  {err:>7.2f}%"
                print(row)


def main():
    project_root = Path(__file__).resolve().parents[3]
    default_csv = project_root / "output" / "convergence" / "results.csv"
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_csv

    if not csv_path.exists():
        print(f"Results not found: {csv_path}")
        print("Run scripts/reproduce_paper/convergence/1_run_convergence.py first.")
        return

    rates = load_rates(csv_path)
    print(f"Loaded {len(rates)} rates from {csv_path}")

    tables = []
    for spec in TABLES:
        table, cells = make_table(rates, spec)
        tables.append(table)
        print_summary(spec, cells)

    out_path = project_root / "output" / "convergence" / "convergence_tables.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "% Convergence tables (tab:conv2_*), generated by 2_make_tables.py\n"
        "% Requires \\usepackage{multirow}\n\n" + "\n\n\n".join(tables) + "\n"
    )
    print(f"\nLaTeX tables -> {out_path}")


if __name__ == "__main__":
    main()
