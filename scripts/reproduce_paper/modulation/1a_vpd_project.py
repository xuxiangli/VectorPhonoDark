"""
Project the SHM VDF and the Al2O3 (light dark photon) material form factor
onto the wavelet-harmonic basis, and store both to HDF5.

Pipeline:   1a_vpd_project.py  ->  1b_vpd_rate.py  ->  2_pd.py  ->  3_plot.py
Output:     output/modulation/vdf.hdf5
            output/modulation/ff.hdf5

Run (vectorphonodark environment):
    python -u scripts/reproduce_paper/modulation/1a_vpd_project.py
"""

import os

# Pin BLAS to one thread before numpy loads: the form-factor projection is
# dominated by tiny per-q-point eigensolves, which multithreaded BLAS makes
# slower. An explicit setting in the environment still wins over this default.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging
import time
from pathlib import Path

import numba
import numpy as np
from scipy import special

from vectorphonodark import constants as const
from vectorphonodark import phonopy_funcs
from vectorphonodark.projection import VDF, FormFactor

# eV; bins start here, 20 meV is extracted downstream in 3_plot.py.
# Must match 2_pd.py.
ENERGY_THRESHOLD = 1e-3
ENERGY_BIN_WIDTH = 1e-3  # eV
ENERGY_MAX_FACTOR = 1.2

L_MAX = 5
# 127  -> N_v = 128 VDF radial wavelets
NV_MAX = 2**7 - 1
# 511  -> N_q = 512 FF radial wavelets
NQ_MAX = 2**9 - 1
# (n_r, n_theta, n_phi) integration grid
VDF_GRID = (128, 180, 180)
FF_GRID = (512, 25, 25)

VDF_HDF5 = "vdf.hdf5"
VDF_GROUP = "SHM"
FF_HDF5 = "ff.hdf5"
FF_GROUP = "q_cut"


@numba.njit
def vdf_shm(v_xyz, v_0, v_e, v_esc, n0) -> float:
    """Standard Halo Model velocity distribution in the lab frame."""
    v_gal = np.linalg.norm(v_xyz + v_e)
    if v_gal <= v_esc:
        return np.exp(-(v_gal**2) / v_0**2) / n0
    return 0.0


def project_vdf() -> VDF:
    """Project the SHM VDF onto the wavelet-harmonic basis (single projection)."""
    v_0, v_esc = const.V0, const.VESC
    # Velocity frame: +z along the DM wind axis at t = 0, so v_E lies on the z-axis.
    v_e = np.array([0.0, 0.0, const.VE])
    n0 = (
        np.pi ** (3 / 2)
        * v_0**2
        * (
            v_0 * special.erf(v_esc / v_0)
            - 2 * v_esc / np.sqrt(np.pi) * np.exp(-(v_esc**2) / v_0**2)
        )
    )
    vdf = VDF(
        physics_params={
            "vdf": vdf_shm,
            "vdf_params": {"v_0": v_0, "v_e": v_e, "v_esc": v_esc, "n0": n0},
            "model": "SHM",
        },
        numerics_params={
            "v_max": const.VESC + const.VE,
            "l_max": L_MAX,
            "n_max": NV_MAX,
            "n_grid": VDF_GRID,
        },
    )
    vdf.project(params={"n_grid": VDF_GRID})
    return vdf


def project_ff(q_cut: float) -> FormFactor:
    """Project the Al2O3 material form factor once: q_max = q_cut, omega_min = 1 meV."""
    root = Path(__file__).resolve().parents[3]
    physics_params = {
        "energy_threshold": ENERGY_THRESHOLD,
        "energy_bin_width": ENERGY_BIN_WIDTH,
        "energy_max_factor": ENERGY_MAX_FACTOR,
        "model": "Al2O3_dark_photon",
    }
    numerics_params = {
        "q_max": q_cut,
        "l_max": L_MAX,
        "n_max": NQ_MAX,
        "n_grid": FF_GRID,
        "log_wavelet": True,
    }
    ff = FormFactor(physics_params=physics_params, numerics_params=numerics_params)
    ff.project(
        params={
            **physics_params,
            **numerics_params,
            "material_input": str(root / "inputs/material/Al2O3/Al2O3.py"),
            "physics_model_input": str(root / "inputs/physics_model/dark_photon.py"),
            "numerics_input": str(root / "inputs/numerics/standard.py"),
        }
    )
    return ff


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = Path(__file__).resolve().parents[3]
    out_dir = root / "output" / "modulation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Al2O3 Debye-Waller momentum cutoff
    q_cut = phonopy_funcs.compute_q_cut(str(root / "inputs/material/Al2O3/Al2O3.py"))

    print("=" * 70)
    print("VectorPhonoDark - Al2O3 light dark photon: VDF + form factor projection")
    print(f"  q_cut = {q_cut:.6e} eV")
    print("=" * 70)

    print("\n[1/2] Projecting SHM VDF ...", flush=True)
    t0 = time.time()
    vdf = project_vdf()
    print(f"      VDF f_lm_n {vdf.f_lm_n.shape}  ({time.time() - t0:.1f} s)")
    vdf.export_hdf5(
        filename=str(out_dir / VDF_HDF5), groupname=VDF_GROUP, dataname="data"
    )

    print(
        "\n[2/2] Projecting material form factor (q_max = q_cut, 1 meV bins) ...",
        flush=True,
    )
    t0 = time.time()
    ff = project_ff(q_cut)
    print(f"      FF: {ff.n_bins} energy bins  ({(time.time() - t0) / 60:.1f} min)")
    ff.export_hdf5(filename=str(out_dir / FF_HDF5), groupname=FF_GROUP, dataname="data")

    print(f"\nStored projections -> {out_dir / VDF_HDF5}, {out_dir / FF_HDF5}")
    print("Next: 1b_vpd_rate.py.")


if __name__ == "__main__":
    main()
