"""
Daily modulation of the DM-phonon scattering rate in Al2O3 (light dark photon),
contracted from a pre-computed VDF and material form factor.

Pipeline:   1a_vpd_project.py  ->  1b_vpd_rate.py  ->  2_pd.py  ->  3_plot.py
Input:      output/modulation/vdf.hdf5, output/modulation/ff.hdf5
Output:     output/modulation/vpd_binned.json

Run (vectorphonodark environment):
    python -u scripts/reproduce_paper/modulation/1b_vpd_rate.py
"""

import json
import logging
import time
from pathlib import Path

import numpy as np

from vectorphonodark import Rotation
from vectorphonodark import constants as const
from vectorphonodark.projection import VDF, FormFactor
from vectorphonodark.rate import Rate

# DM masses (keV)
MASSES_KEV = [50, 100, 500, 1000]
# hourly steps over one day
N_TIMES = 24

L_MAX = 5
NV_MAX = 2**7 - 1
NQ_MAX = 2**9 - 1
VDF_GRID = (128, 180, 180)
FF_GRID = (512, 25, 25)

# Light dark photon:  F_DM = (q0/q)^2  ->  fdm = (-4, 0)
# with q0 = Q_BOHR = alpha m_e
FDM = (-4, 0)
Q0_FDM = const.Q_BOHR
MASS_SM = const.M_ELEC

VDF_HDF5 = "vdf.hdf5"
VDF_GROUP = "SHM"
FF_HDF5 = "ff.hdf5"
FF_GROUP = "q_cut"

OUT_JSON = "vpd_binned.json"


def build_rotations(n_times: int) -> list[Rotation]:
    """Crystal rotations at hourly steps: rotations about the Earth's spin
    axis."""
    axis = np.array([0.0, -np.sin(const.THETA_E), np.cos(const.THETA_E)])
    return [
        Rotation.from_axis_angle(axis, 2.0 * np.pi * i / n_times)
        for i in range(n_times)
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = Path(__file__).resolve().parents[3]
    out_dir = root / "output" / "modulation"

    vdf_path = out_dir / VDF_HDF5
    ff_path = out_dir / FF_HDF5
    if not vdf_path.is_file() or not ff_path.is_file():
        raise SystemExit(
            f"Missing {vdf_path if not vdf_path.is_file() else ff_path}: "
            "run 1a_vpd_project.py first."
        )

    print("=" * 70)
    print("VectorPhonoDark - Al2O3 light dark photon, daily modulation (1 meV bins)")
    print(f"  masses = {MASSES_KEV} keV,  {N_TIMES} hourly steps")
    print("=" * 70)

    print(f"\nLoading VDF <- {vdf_path}", flush=True)
    vdf = VDF().import_hdf5(filename=str(vdf_path), groupname=VDF_GROUP)
    print(f"Loading form factor <- {ff_path}", flush=True)
    ff = FormFactor().import_hdf5(filename=str(ff_path), groupname=FF_GROUP)
    print(f"  q_max = {ff.q_max:.6e} eV, {ff.n_bins} energy bins")

    t_total0 = time.time()
    rotations = build_rotations(N_TIMES)

    print("\nRate contraction + daily modulation ...", flush=True)
    rate_response = {}
    for mass_keV in MASSES_KEV:
        t0 = time.time()
        rate = Rate(
            physics_params={
                "fdm": FDM,
                "q0_fdm": Q0_FDM,
                "mass_dm": mass_keV * 1e3,
                "mass_sm": MASS_SM,
            },
            numerics_params={"l_max": L_MAX, "nv_max": NV_MAX, "nq_max": NQ_MAX},
            vdf=vdf,
            ff=ff,
        )
        mu_R = rate.binned_rate(rotations=rotations)  # {bin: (n_times,)}
        binned = np.zeros((ff.n_bins, N_TIMES))
        for b, arr in mu_R.items():
            binned[b] = np.asarray(arr, dtype=float)
        rate_response[mass_keV] = binned
        print(f"      m_chi = {mass_keV:>4} keV   ({time.time() - t0:.1f} s)")
    print(f"  total rate computation time: {time.time() - t_total0:.1f} s")

    thr, width = float(ff.energy_threshold), float(ff.energy_bin_width)
    payload = {
        "code": "VectorPhonoDark",
        "material": "Al2O3",
        "model": "light_dark_photon",
        "masses_keV": MASSES_KEV,
        "n_times": N_TIMES,
        "energy_threshold_meV": thr * 1e3,
        "energy_bin_width_meV": width * 1e3,
        "bin_lower_edges_meV": [(thr + b * width) * 1e3 for b in range(ff.n_bins)],
        "q_max_eV": float(ff.q_max),
        "fdm": list(FDM),
        "q0_fdm_eV": Q0_FDM,
        "mass_sm_eV": MASS_SM,
        "params": {
            "l_max": L_MAX,
            "nv_max": NV_MAX,
            "nq_max": NQ_MAX,
            "vdf_grid": list(VDF_GRID),
            "ff_grid": list(FF_GRID),
            "log_wavelet": True,
        },
        # rate_response[mass] has shape (n_bins, n_times): the per-bin rate
        # response Gamma_b(t) in natural units (eV^-2). 3_plot.py sums the bins
        # above any threshold and converts to rate.
        "rate_response": {str(m): rate_response[m].tolist() for m in MASSES_KEV},
    }
    out_path = out_dir / OUT_JSON
    with open(out_path, "w") as fh:
        json.dump(payload, fh)
    print(f"\nStored energy-binned rate response -> {out_path}")
    print("Next: 2_pd.py (PhonoDark), then 3_plot.py.")


if __name__ == "__main__":
    main()
