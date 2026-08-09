"""Project the inputs for the bin-width convergence figure (both materials).

For the draft figure fig:convergence_binwidth: the SHM VDF, and one form
factor per (material, coupling, bin width, mass):

  materials   GaAs, Al2O3
  couplings   hadrophilic, dark photon (the hadrophilic projection serves
              both the heavy and light models; F_DM enters only at rate time)
  bin widths  1.0, 0.5, 0.25, 0.1 meV
  masses      10 .. 100 keV, each on its own kinematic domain 2*m_chi*v_max

Each HDF5 file holds one (material, coupling, bin width), with one group per
mass. Existing groups are skipped, so an interrupted run resumes.

Output: output/binwidth/

Usage:
    python scripts/reproduce_paper/binwidth/1_project_ff.py
"""

import os

# Pin BLAS to one thread before numpy loads: the form-factor projection is
# dominated by tiny per-q-point eigensolves, which multithreaded BLAS makes
# slower. An explicit setting in the environment still wins over this default.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import time
from pathlib import Path

import h5py
import numba
import numpy as np
from scipy import special

from vectorphonodark import constants as const
from vectorphonodark.projection import VDF, FormFactor

# =====================  Configuration  =====================

MATERIALS = ["GaAs", "Al2O3"]
FF_MODELS = ["hadrophilic", "dark_photon"]
MASSES_KEV = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
BIN_WIDTHS_EV = [1e-3, 5e-4, 2.5e-4, 1e-4]

ENERGY_THRESHOLD = 1e-3  # eV
L_MAX = 5
NQ_MAX = 511
N_GRID_FF = (512, 25, 25)

NV_MAX = 127
N_GRID_VDF = (128, 180, 180)

VDF_GROUP = "SHM/t0"
HDF5_DATA = "data"

DE_LABEL = {1e-3: "1p0meV", 5e-4: "0p5meV", 2.5e-4: "0p25meV", 1e-4: "0p1meV"}


# =====================  SHM helpers  =====================


@numba.njit
def vdf_shm(v_xyz, v_0, v_e, v_esc, n0) -> float:
    v_gal_frame = np.linalg.norm(v_xyz + v_e)
    if v_gal_frame <= v_esc:
        return np.exp(-(v_gal_frame**2) / v_0**2) / n0
    return 0.0


def _shm_params():
    v_0 = const.V0
    # Velocity frame: +z along the DM wind axis at t = 0, so v_E lies on the z-axis.
    v_e = np.array([0.0, 0.0, const.VE])
    v_esc = const.VESC
    n0 = (
        np.pi ** (3 / 2)
        * v_0**2
        * (
            v_0 * special.erf(v_esc / v_0)
            - 2 * v_esc / np.sqrt(np.pi) * np.exp(-((v_esc / v_0) ** 2))
        )
    )
    return {"v_0": v_0, "v_e": v_e, "v_esc": v_esc, "n0": n0}


# =====================  Projections  =====================


def project_vdf(output_dir):
    path = output_dir / "vdf.hdf5"
    if path.exists():
        print("  [skip] VDF (file already exists)")
        return
    print(f"  VDF: l_max={L_MAX}, nv_max={NV_MAX}, n_grid={N_GRID_VDF}")
    shm = _shm_params()
    pp = {"vdf": vdf_shm, "vdf_params": shm, "model": "SHM"}
    np_ = {
        "v_max": const.VESC + const.VE,
        "l_max": L_MAX,
        "n_max": NV_MAX,
        "n_grid": N_GRID_VDF,
    }
    vdf = VDF(physics_params=pp, numerics_params=np_)
    vdf.project(params={**pp, **np_})
    vdf.export_hdf5(filename=str(path), groupname=VDF_GROUP, dataname=HDF5_DATA)
    print(f"    saved -> {path}")


def group_exists(hdf5_path, group_name):
    if not Path(hdf5_path).exists():
        return False
    with h5py.File(hdf5_path, "r") as f:
        return "log" in f and group_name in f["log"]


def project_one(project_root, hdf5_path, material, mass_eV, dE, ff_model):
    q_max = 2 * mass_eV * (const.VESC + const.VE)
    group_name = f"{q_max} eV"
    if group_exists(hdf5_path, group_name):
        print(" already exists, skip.")
        return

    physics_params = {
        "energy_threshold": ENERGY_THRESHOLD,
        "energy_bin_width": dE,
        "energy_max_factor": 1.2,
        "model": f"{material}_{ff_model}",
    }
    numerics_params = {
        "q_max": q_max,
        "l_max": L_MAX,
        "n_max": NQ_MAX,
        "n_grid": N_GRID_FF,
        "log_wavelet": True,
    }
    params = {
        **physics_params,
        **numerics_params,
        "material_input": str(
            project_root / "inputs" / "material" / material / f"{material}.py"
        ),
        "physics_model_input": str(
            project_root / "inputs" / "physics_model" / f"{ff_model}.py"
        ),
        "numerics_input": str(project_root / "inputs" / "numerics" / "standard.py"),
    }

    ff = FormFactor(physics_params=physics_params, numerics_params=numerics_params)
    ff.project(params=params)
    ff.export_hdf5(
        filename=hdf5_path, groupname=f"log/{group_name}", dataname=HDF5_DATA
    )
    print(" done.")


# =====================  Main  =====================


def main():
    project_root = Path(__file__).resolve().parents[3]
    output_dir = project_root / "output" / "binwidth"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory : {output_dir}")

    print("\n=== Projecting VDF ===")
    project_vdf(output_dir)

    t_total = time.time()
    for material in MATERIALS:
        for ff_model in FF_MODELS:
            print(f"\n{'=' * 60}")
            print(f"{material} / {ff_model}")
            print(f"{'=' * 60}")
            for dE in BIN_WIDTHS_EV:
                hdf5_path = str(
                    output_dir / f"{material}_{ff_model}_{DE_LABEL[dE]}.hdf5"
                )
                print(f"\n  dE = {dE * 1e3:.2f} meV  ->  {Path(hdf5_path).name}")
                for mass_keV in MASSES_KEV:
                    t0 = time.time()
                    print(f"    m = {mass_keV} keV ...", end="", flush=True)
                    project_one(
                        project_root, hdf5_path, material, mass_keV * 1e3, dE, ff_model
                    )
                    print(f"    ({time.time() - t0:.1f} s)")

    print(f"\nAll done in {(time.time() - t_total) / 60:.1f} min.")


if __name__ == "__main__":
    main()
