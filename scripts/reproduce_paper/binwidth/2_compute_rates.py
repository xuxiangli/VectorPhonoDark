"""Compute the rate at every bin width for the bin-width convergence figure.

Reads the projections from 1_project_ff.py and computes the total rate
(events per kg-yr at sigma_0_bar = 1 cm^2) for three DM models x two
materials x ten masses x four bin widths. The hadrophilic form factor
serves both the heavy and light models; F_DM enters only through the
kinematic kernel.

Results land in output/binwidth/results.json, saved incrementally so an
interrupted run resumes. A convergence summary against the 0.1 meV
reference is printed at the end; 3_plot.py makes the figures from the JSON.

Usage:
    python scripts/reproduce_paper/binwidth/2_compute_rates.py
"""

import json
import time
from pathlib import Path

import numpy as np

from vectorphonodark import Rotation
from vectorphonodark import constants as const
from vectorphonodark.projection import VDF, FormFactor
from vectorphonodark.rate import Rate

# =====================  Configuration  =====================

MATERIALS = ["GaAs", "Al2O3"]
MASSES_KEV = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
BIN_WIDTHS_EV = [1e-3, 5e-4, 2.5e-4, 1e-4]
# reference cross section for the reported rate (cm^2)
SIGMA_0_BAR = 1.0
L_MAX = 5
NV_MAX = 127
NQ_MAX = 511

MODELS = {
    "hadrophilic_heavy": {
        "fdm": (0, 0),
        "mass_sm": const.M_NUCL,
        "q0_fdm_func": lambda m: m * const.V0,
        "ff_model": "hadrophilic",
    },
    "hadrophilic_light": {
        "fdm": (-4, 0),
        "mass_sm": const.M_NUCL,
        "q0_fdm_func": lambda m: m * const.V0,
        "ff_model": "hadrophilic",
    },
    "dark_photon_light": {
        "fdm": (-4, 0),
        "mass_sm": const.M_ELEC,
        "q0_fdm_func": lambda _m: const.Q_BOHR,
        "ff_model": "dark_photon",
    },
}

DE_LABEL = {1e-3: "1p0meV", 5e-4: "0p5meV", 2.5e-4: "0p25meV", 1e-4: "0p1meV"}

VDF_GROUP = "SHM/t0"
HDF5_DATA = "data"

# No rotation acting on the crystal
ROTATIONS = [Rotation.identity()]


# =====================  Rates  =====================


def load_ff(output_dir, material, ff_model, dE_eV, mass_eV):
    q_max = 2 * mass_eV * (const.VESC + const.VE)
    hdf5_path = str(output_dir / f"{material}_{ff_model}_{DE_LABEL[dE_eV]}.hdf5")
    return FormFactor().import_hdf5(
        filename=hdf5_path, groupname=f"log/{q_max} eV", dataname=HDF5_DATA
    )


def compute_rate(vdf, ff, physics_params):
    """Total rate in events per kg-yr at cross section SIGMA_0_BAR."""
    rate_obj = Rate(
        physics_params=physics_params,
        numerics_params={"l_max": L_MAX, "nv_max": NV_MAX, "nq_max": NQ_MAX},
        vdf=vdf,
        ff=ff,
    )
    total_rate = float(
        sum(v[0] for v in rate_obj.binned_rate(rotations=ROTATIONS).values())
    )
    return SIGMA_0_BAR * const.KG_YR * const.RHO_DM * total_rate / const.INVEV_TO_CM**2


# =====================  Main  =====================


def main():
    project_root = Path(__file__).resolve().parents[3]
    output_dir = project_root / "output" / "binwidth"
    results_file = output_dir / "results.json"

    vdf_path = output_dir / "vdf.hdf5"
    if not vdf_path.exists():
        print(f"VDF not found: {vdf_path}")
        print("Run scripts/reproduce_paper/binwidth/1_project_ff.py first.")
        return
    vdf = VDF().import_hdf5(
        filename=str(vdf_path), groupname=VDF_GROUP, dataname=HDF5_DATA
    )

    # results[material][model][mass_label][dE_label] = rate at SIGMA_0_BAR
    # (events per kg-yr)
    results = {}
    if results_file.exists():
        results = json.loads(results_file.read_text())
        print(f"Loaded partial results from {results_file}")

    n_total = len(MATERIALS) * len(MODELS) * len(MASSES_KEV) * len(BIN_WIDTHS_EV)
    n_done = 0

    t_wall = time.time()
    for material in MATERIALS:
        mat_results = results.setdefault(material, {})
        for model_key, cfg in MODELS.items():
            model_results = mat_results.setdefault(model_key, {})
            print(f"\n  {material} / {model_key}")
            for mass_keV in MASSES_KEV:
                mass_eV = mass_keV * 1e3
                mk = f"{mass_keV}keV"
                mass_results = model_results.setdefault(mk, {})
                pp = {
                    "fdm": cfg["fdm"],
                    "q0_fdm": cfg["q0_fdm_func"](mass_eV),
                    "mass_dm": mass_eV,
                    "mass_sm": cfg["mass_sm"],
                }
                for dE_eV in BIN_WIDTHS_EV:
                    dE_label = DE_LABEL[dE_eV]
                    n_done += 1
                    if dE_label in mass_results:
                        print(f"    [{n_done:3d}/{n_total}] {mk:>7s} {dE_label:>8s}")
                        continue
                    # Start the line now so the log shows what is running
                    print(
                        f"    [{n_done:3d}/{n_total}] {mk:>7s} {dE_label:>8s} ... ",
                        end="",
                        flush=True,
                    )
                    t0 = time.time()
                    ff = load_ff(output_dir, material, cfg["ff_model"], dE_eV, mass_eV)
                    rate = compute_rate(vdf, ff, pp)
                    mass_results[dE_label] = rate
                    print(f"({time.time() - t0:.1f} s)")
                # Incremental save after each mass
                results_file.write_text(json.dumps(results, indent=2))

    print(f"\nSaved to {results_file}")
    print(f"Total elapsed: {(time.time() - t_wall) / 60:.1f} min.")

    # Convergence summary against the finest bin width
    ref_label = DE_LABEL[BIN_WIDTHS_EV[-1]]
    print("\n" + "=" * 65)
    print(f"Relative rate error vs {BIN_WIDTHS_EV[-1] * 1e3:g} meV reference")
    print("=" * 65)
    for material in MATERIALS:
        for model_key in MODELS:
            print(f"\n  {material} / {model_key}")
            header = f"  {'mass':>8s}" + "".join(
                f"  {DE_LABEL[d]:>10s}" for d in BIN_WIDTHS_EV[:-1]
            )
            print(header)
            for mass_keV in MASSES_KEV:
                d = results[material][model_key][f"{mass_keV}keV"]
                ref = d.get(ref_label, float("nan"))
                row = f"  {f'{mass_keV}keV':>8s}"
                for dE_eV in BIN_WIDTHS_EV[:-1]:
                    val = d.get(DE_LABEL[dE_eV], float("nan"))
                    if ref > 0 and not (np.isinf(ref) or np.isinf(val)):
                        row += f"  {(val - ref) / ref * 100:>+9.2f}%"
                    else:
                        row += f"  {'nan':>10s}"
                print(row)


if __name__ == "__main__":
    main()
