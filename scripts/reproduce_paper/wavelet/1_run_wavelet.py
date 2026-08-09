"""Run the wavelet convergence study: linear vs log Haar basis (GaAs).

For the draft figures fig:wavelet_comparison_{light,heavy}: rates for
  - DM mass:      0.1, 1, 10, 100 MeV
  - basis:        linear and log wavelets
  - mediator:     light (F_DM^2 ~ q^-4) and heavy (F_DM^2 = 1) hadrophilic
  - N_q:          1, 2, 4, ..., 2048  (nq_max = N_q - 1)

The material form factor is projected once per (mass, basis) at N_q = 2048 on its own
kinematic domain q_max = min(2 m_chi v_max, q_cut); every smaller N_q is a
truncation of that projection at rate time. The VDF is projected once
(l_max = 5, N_v = 128). Results go to output/wavelet/results.csv.

Usage:
    python scripts/reproduce_paper/wavelet/1_run_wavelet.py
"""

import os

# Pin BLAS to one thread before numpy loads: the form-factor projection is
# dominated by tiny per-q-point eigensolves, which multithreaded BLAS makes
# slower. An explicit setting in the environment still wins over this default.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import csv
from pathlib import Path

import h5py
import numba
import numpy as np
from scipy import special

from vectorphonodark import Rotation, phonopy_funcs
from vectorphonodark import constants as const
from vectorphonodark.projection import VDF, FormFactor
from vectorphonodark.rate import Rate

# =====================  Configuration  =====================

MATERIAL = "GaAs"
PHYSICS_MODEL = "hadrophilic.py"

MASSES_EV = [0.1e6, 1.0e6, 10.0e6, 100.0e6]

MEDIATOR_MODELS = {
    "light": {"fdm": (-4, 0), "q0_fdm_func": lambda m: m * const.V0},
    "heavy": {"fdm": (0, 0), "q0_fdm_func": lambda m: m * const.V0},
}

# nq_max = 2^k - 1, so each truncation keeps N_q = 2^k wavelets.
NQ_MAX_LIST = [2**k - 1 for k in range(12)]

# Form-factor projection: the finest truncation, on its minimal radial grid.
L_MAX = 5
NQ_MAX_FF = 2047
N_GRID_FF = (2048, 25, 25)

# VDF projection, N_v = 128
NV_MAX = 127
N_GRID_VDF = (128, 180, 180)

VDF_HDF5_GROUP = "SHM/t0"
HDF5_DATA = "data"

# No rotation acting on the crystal
ROTATIONS = [Rotation.identity()]


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


# =====================  Path helpers  =====================


def material_input(project_root):
    return str(project_root / "inputs" / "material" / MATERIAL / f"{MATERIAL}.py")


def _wc_dir(output_dir):
    return output_dir / "wavelet"


def vdf_hdf5_path(output_dir):
    return _wc_dir(output_dir) / "vdf.hdf5"


def ff_hdf5_path(output_dir, wavelet_type):
    return _wc_dir(output_dir) / f"ff_{wavelet_type}.hdf5"


def q_max_for_mass(mass_ev, q_cutoff):
    return min(2.0 * mass_ev * (const.VESC + const.VE), q_cutoff)


def ff_hdf5_group(mass_ev, q_cutoff):
    return f"{q_max_for_mass(mass_ev, q_cutoff)} eV"


# =====================  Projections  =====================


def project_vdf(output_dir):
    path = vdf_hdf5_path(output_dir)
    if path.exists():
        print("  [skip] VDF (file already exists)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
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
    vdf.export_hdf5(filename=str(path), groupname=VDF_HDF5_GROUP, dataname=HDF5_DATA)
    print(f"    saved -> {path}")


def project_ff(project_root, output_dir, mass_ev, wavelet_type, q_cutoff):
    path = ff_hdf5_path(output_dir, wavelet_type)
    group = ff_hdf5_group(mass_ev, q_cutoff)

    if path.exists():
        with h5py.File(str(path), "r") as fh:
            if group in fh:
                print(f"  [skip] FF {wavelet_type} / {mass_ev / 1e6:.1f} MeV")
                return
    path.parent.mkdir(parents=True, exist_ok=True)

    q_max = q_max_for_mass(mass_ev, q_cutoff)
    print(
        f"  FF {wavelet_type} / {mass_ev / 1e6:.1f} MeV  "
        f"q_max={q_max:.4e} eV  nq_max={NQ_MAX_FF}  n_grid={N_GRID_FF}"
    )

    physics_params = {
        "energy_threshold": 1e-3,
        "energy_bin_width": 1e-3,
        "energy_max_factor": 1.2,
        "model": f"{MATERIAL}_hadrophilic",
    }
    numerics_params = {
        "q_max": q_max,
        "l_max": L_MAX,
        "n_max": NQ_MAX_FF,
        "n_grid": N_GRID_FF,
        "log_wavelet": wavelet_type == "log",
    }
    input_params = {
        "material_input": material_input(project_root),
        "physics_model_input": str(
            project_root / "inputs" / "physics_model" / PHYSICS_MODEL
        ),
        "numerics_input": str(project_root / "inputs" / "numerics" / "standard.py"),
    }

    ff = FormFactor(physics_params=physics_params, numerics_params=numerics_params)
    ff.project(params={**physics_params, **numerics_params, **input_params})
    ff.export_hdf5(filename=str(path), groupname=group, dataname=HDF5_DATA)
    print(f"    saved -> {path} [{group}]")


# =====================  Rates  =====================


def compute_rates_for_ff(vdf_obj, ff_obj, mass_ev, mediator_type):
    """Return {nq_max: rate}, truncating the N_q = 2048 projection."""
    model = MEDIATOR_MODELS[mediator_type]
    rates = {}
    for nq_max in NQ_MAX_LIST:
        physics_params = {
            "fdm": model["fdm"],
            "q0_fdm": model["q0_fdm_func"](mass_ev),
            "mass_dm": mass_ev,
            "mass_sm": const.M_NUCL,
        }
        numerics_params = {"l_max": L_MAX, "nv_max": NV_MAX, "nq_max": nq_max}
        rate_obj = Rate(
            physics_params=physics_params,
            numerics_params=numerics_params,
            vdf=vdf_obj,
            ff=ff_obj,
        )
        rate_r = sum(v for v in rate_obj.binned_rate(rotations=ROTATIONS).values())
        rates[nq_max] = float(rate_r[0])
        print(f"    nq_max={nq_max:5d}: rate = {rates[nq_max]:.6e}")
    return rates


def _warmup():
    shm = _shm_params()
    pp = {"vdf": vdf_shm, "vdf_params": shm, "model": "SHM"}
    np_ = {"v_max": const.VESC + const.VE, "l_max": 1, "n_max": 3, "n_grid": (4, 4, 4)}
    v = VDF(physics_params=pp, numerics_params=np_)
    v.project(params={**pp, **np_})
    print("  Numba JIT warm-up complete.")


# =====================  Main  =====================


def main():
    project_root = Path(__file__).resolve().parents[3]
    output_dir = project_root / "output"
    _wc_dir(output_dir).mkdir(parents=True, exist_ok=True)

    print("Warming up Numba JIT ...")
    _warmup()

    q_cutoff = phonopy_funcs.compute_q_cut(material_input(project_root))
    print(f"\nq_cutoff[{MATERIAL}] = {q_cutoff:.6e} eV")

    # --- 1. Project VDF ---
    print("\n=== Projecting VDF ===")
    project_vdf(output_dir)

    # --- 2. Project material form factors (8 = 2 bases x 4 masses) ---
    print("\n=== Projecting material form factors ===")
    for wavelet_type in ("linear", "log"):
        for mass_ev in MASSES_EV:
            project_ff(project_root, output_dir, mass_ev, wavelet_type, q_cutoff)

    # --- 3. Rates ---
    vdf_obj = VDF().import_hdf5(
        filename=str(vdf_hdf5_path(output_dir)),
        groupname=VDF_HDF5_GROUP,
        dataname=HDF5_DATA,
    )

    print("\n=== Computing rates ===")
    results = []
    for wavelet_type in ("linear", "log"):
        for mass_ev in MASSES_EV:
            ff_obj = FormFactor().import_hdf5(
                filename=str(ff_hdf5_path(output_dir, wavelet_type)),
                groupname=ff_hdf5_group(mass_ev, q_cutoff),
                dataname=HDF5_DATA,
            )
            for mediator_type in ("light", "heavy"):
                print(
                    f"\n  === {wavelet_type}/{mediator_type}/"
                    f"{mass_ev / 1e6:.1f} MeV ==="
                )
                rates = compute_rates_for_ff(vdf_obj, ff_obj, mass_ev, mediator_type)
                results.extend(
                    {
                        "wavelet": wavelet_type,
                        "mediator": mediator_type,
                        "mass_MeV": mass_ev / 1e6,
                        "nq_max": nq_max,
                        "rate": rate,
                    }
                    for nq_max, rate in rates.items()
                )

    # --- 4. Save ---
    csv_path = _wc_dir(output_dir) / "results.csv"
    with open(str(csv_path), "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["wavelet", "mediator", "mass_MeV", "nq_max", "rate"]
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {len(results)} rows to {csv_path}")


if __name__ == "__main__":
    main()
