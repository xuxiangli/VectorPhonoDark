"""Run the q_max reuse study behind the draft's two q_max figures.

A material form factor projected once at the universal q_max = q_cut can be reused
for every DM mass. This study measures what that reuse costs: for each
(material, model, mass), the rate from the universal projection is compared
against a reference rate from a projection at the mass's own kinematic
limit q_max(m_chi) = 2 m_chi (v_esc + v_E).

Projections needed per material:
  - log-basis references (hadrophilic and dark photon), one per mass. The
    top mass saturates at q_cut, so that reference IS the universal projection.
  - linear-basis references at N_q = 128, the basis's own converged heavy
    reference (draft fig:qmax_heavy caption), one per mass below q_cut.
  - one linear-basis projection at N_q = 512 and q_max = q_cut: the universal
    projection.

Output: output/qmax/results.csv, one row per (material, model, mass) whose
q_max(m_chi) is below q_cut, with the reference and universal rates.
2_plot_qmax.py makes the figures. Projections are cached under output/qmax/
and reused, so an interrupted run resumes cheaply.

Usage:
    python scripts/reproduce_paper/qmax/1_run_qmax.py
"""

import os

# Pin BLAS to one thread before numpy loads: the form-factor projection is
# dominated by tiny per-q-point eigensolves, which multithreaded BLAS makes
# slower. An explicit setting in the environment still wins over this default.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import csv
from pathlib import Path

import numba
import numpy as np
from scipy import special

from vectorphonodark import Rotation, phonopy_funcs
from vectorphonodark import constants as const
from vectorphonodark.projection import VDF, FormFactor
from vectorphonodark.rate import Rate

# =====================  Configuration  =====================

MATERIALS = ["GaAs", "Al2O3"]

# Log-spaced masses, 6 per decade. The grid starts at 0.03 MeV.
MASSES_EV = [float(m) for m in np.logspace(np.log10(0.01e6), np.log10(100.0e6), 25)][3:]

# Form-factor projection configurations, keyed by the ff_model label used in
# file paths. The material form factor does not depend on the mediator weight F_DM, so
# the light and heavy hadrophilic models share the hadrophilic projections.
FF_MODELS = {
    "hadrophilic_log": {
        "physics_model": "hadrophilic.py",
        "log_wavelet": True,
        # N_q = 512
        "nq_max": 511,
        "n_grid": (2048, 25, 25),
    },
    "dark_photon_log": {
        "physics_model": "dark_photon.py",
        "log_wavelet": True,
        "nq_max": 511,
        "n_grid": (2048, 25, 25),
    },
    "hadrophilic_linear": {
        "physics_model": "hadrophilic.py",
        "log_wavelet": False,
        # N_q = 128, the linear heavy reference
        "nq_max": 127,
        "n_grid": (2048, 25, 25),
    },
    "hadrophilic_linear_511": {
        "physics_model": "hadrophilic.py",
        "log_wavelet": False,
        "nq_max": 511,
        "n_grid": (2048, 25, 25),
    },
}

# Rate models: `ref_ff` is projected per mass and provides the reference
# rate; `test_ff` is the basis evaluated at the universal q_max = q_cut.
# The linear N_q = 512 universal projection is compared against the linear
# N_q = 128 reference (draft fig:qmax_heavy); the log models test their own
# basis.
MODELS = {
    "light_hadrophilic": {
        "fdm": (-4, 0),
        "mass_sm": const.M_NUCL,
        "q0_fdm_func": lambda m: m * const.V0,
        "ref_ff": "hadrophilic_log",
        "test_ff": "hadrophilic_log",
    },
    "light_dark_photon": {
        "fdm": (-4, 0),
        "mass_sm": const.M_ELEC,
        "q0_fdm_func": lambda _m: const.ALPHA_EM * const.M_ELEC,
        "ref_ff": "dark_photon_log",
        "test_ff": "dark_photon_log",
    },
    "heavy_hadrophilic_log": {
        "fdm": (0, 0),
        "mass_sm": const.M_NUCL,
        "q0_fdm_func": lambda m: m * const.V0,
        "ref_ff": "hadrophilic_log",
        "test_ff": "hadrophilic_log",
    },
    "heavy_hadrophilic_linear": {
        "fdm": (0, 0),
        "mass_sm": const.M_NUCL,
        "q0_fdm_func": lambda m: m * const.V0,
        "ref_ff": "hadrophilic_linear",
        "test_ff": "hadrophilic_linear_511",
    },
}

L_MAX = 5
# N_v = 128
NV_MAX = 127
N_GRID_VDF = (128, 180, 180)

VDF_GROUP = "SHM/t0"
FF_GROUP = "data"
HDF5_DATA = "data"

CSV_FIELDS = [
    "material",
    "model",
    "mass_MeV",
    "q_max_ref",
    "q_cut",
    "rate_ref",
    "rate_universal",
]

# No rotation acting on the crystal
ROTATIONS = [Rotation.identity()]


# =====================  Path helpers  =====================


def _cmp_dir(output_dir):
    return output_dir / "qmax"


def vdf_path(output_dir):
    return _cmp_dir(output_dir) / "vdf.hdf5"


def mass_label(mass_ev):
    return f"mass{mass_ev / 1e6:.4g}MeV"


def material_input(project_root, material):
    return str(project_root / "inputs" / "material" / material / f"{material}.py")


def ref_ff_path(output_dir, material, ff_model, mass_ev):
    return (
        _cmp_dir(output_dir)
        / "ref_ff"
        / material
        / ff_model
        / mass_label(mass_ev)
        / "ff.hdf5"
    )


def q_max_for_mass(mass_ev, q_cutoff):
    """The kinematic limit 2 m_chi v_max, capped at the Debye-Waller cutoff."""
    return min(2.0 * mass_ev * (const.VESC + const.VE), q_cutoff)


def universal_ff_path(output_dir, material, ff_model, q_cutoff):
    """Path of the q_max = q_cut projection for a basis.

    For the reference bases this is the top-mass reference file, whose
    kinematic q_max saturates at q_cut. Only the linear N_q = 512 basis
    needs a file of its own.
    """
    top = MASSES_EV[-1]
    if q_max_for_mass(top, q_cutoff) == q_cutoff and ff_model in {
        m["ref_ff"] for m in MODELS.values()
    }:
        return ref_ff_path(output_dir, material, ff_model, top)
    return (
        _cmp_dir(output_dir) / "test_ff" / material / ff_model / "qcutoff" / "ff.hdf5"
    )


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
    path = vdf_path(output_dir)
    if path.exists():
        print("  [skip] VDF")
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
    v = VDF(physics_params=pp, numerics_params=np_)
    v.project(params={**pp, **np_})
    v.export_hdf5(filename=str(path), groupname=VDF_GROUP, dataname=HDF5_DATA)
    print(f"    saved -> {path}")


def ensure_ff(project_root, dst, material, ff_model_name, q_max):
    """Project the material form factor into `dst`, unless it is already there."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    ff_model = FF_MODELS[ff_model_name]
    print(
        f"  FF {material}/{ff_model_name}  q_max={q_max:.4e}  "
        f"nq_max={ff_model['nq_max']}  n_grid={ff_model['n_grid']}",
        flush=True,
    )
    pp = {
        "energy_threshold": 1e-3,
        "energy_bin_width": 1e-3,
        "energy_max_factor": 1.2,
        "model": f"{material}_{ff_model_name}",
    }
    np_ = {
        "q_max": q_max,
        "l_max": L_MAX,
        "n_max": ff_model["nq_max"],
        "n_grid": ff_model["n_grid"],
        "log_wavelet": ff_model["log_wavelet"],
    }
    ip = {
        "material_input": material_input(project_root, material),
        "physics_model_input": str(
            project_root / "inputs" / "physics_model" / ff_model["physics_model"]
        ),
        "numerics_input": str(project_root / "inputs" / "numerics" / "standard.py"),
    }
    ff = FormFactor(physics_params=pp, numerics_params=np_)
    ff.project(params={**pp, **np_, **ip})
    ff.export_hdf5(filename=str(dst), groupname=FF_GROUP, dataname=HDF5_DATA)
    print(f"    saved -> {dst.parent.name}/{dst.name}")


def load_ff(path):
    return FormFactor().import_hdf5(
        filename=str(path), groupname=FF_GROUP, dataname=HDF5_DATA
    )


# =====================  Rates  =====================


def compute_rate(vdf_obj, ff, model_name, ff_model_name, mass_ev):
    model = MODELS[model_name]
    pp = {
        "fdm": model["fdm"],
        "q0_fdm": model["q0_fdm_func"](mass_ev),
        "mass_dm": mass_ev,
        "mass_sm": model["mass_sm"],
    }
    np_ = {
        "l_max": L_MAX,
        "nv_max": NV_MAX,
        "nq_max": FF_MODELS[ff_model_name]["nq_max"],
    }
    rate = Rate(physics_params=pp, numerics_params=np_, vdf=vdf_obj, ff=ff)
    return float(sum(rate.binned_rate(ROTATIONS).values())[0])


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
    cmp_dir = _cmp_dir(output_dir)
    cmp_dir.mkdir(parents=True, exist_ok=True)

    print("Warming up Numba JIT ...")
    _warmup()

    # The Debye-Waller cutoff
    q_cutoff = {
        material: phonopy_funcs.compute_q_cut(material_input(project_root, material))
        for material in MATERIALS
    }
    for material, q_cut in q_cutoff.items():
        print(f"  q_cutoff[{material}] = {q_cut:.6e} eV")

    # --- 1. VDF, projected once and shared by every rate below ---
    print("\n=== Projecting VDF ===")
    project_vdf(output_dir)
    vdf_obj = VDF().import_hdf5(
        filename=str(vdf_path(output_dir)), groupname=VDF_GROUP, dataname=HDF5_DATA
    )

    # --- 2. Material form factors ---
    ref_ff_models = {model["ref_ff"] for model in MODELS.values()}
    print("\n=== Ensuring reference material form factors ===")
    for material in MATERIALS:
        for ff_model_name in sorted(ref_ff_models):
            for mass_ev in MASSES_EV:
                q_max = q_max_for_mass(mass_ev, q_cutoff[material])
                if (
                    q_max == q_cutoff[material]
                    and not FF_MODELS[ff_model_name]["log_wavelet"]
                ):
                    continue
                ensure_ff(
                    project_root,
                    ref_ff_path(output_dir, material, ff_model_name, mass_ev),
                    material,
                    ff_model_name,
                    q_max,
                )

    print("\n=== Ensuring universal (q_max = q_cut) material form factors ===")
    for material in MATERIALS:
        for ff_model_name in sorted({m["test_ff"] for m in MODELS.values()}):
            path = universal_ff_path(
                output_dir, material, ff_model_name, q_cutoff[material]
            )
            if path.exists():
                print(f"  [have] {material}/{ff_model_name} -> {path.parent.name}")
                continue
            ensure_ff(project_root, path, material, ff_model_name, q_cutoff[material])

    # --- 3. Rates ---
    print("\n=== Computing rates ===")
    results = []
    for material in MATERIALS:
        q_cut = q_cutoff[material]
        for model_name, model in MODELS.items():
            universal_ff = load_ff(
                universal_ff_path(output_dir, material, model["test_ff"], q_cut)
            )
            for mass_ev in MASSES_EV:
                q_max_ref = q_max_for_mass(mass_ev, q_cut)
                if q_max_ref == q_cut:
                    continue  # the reference is the universal projection
                ref_ff = load_ff(
                    ref_ff_path(output_dir, material, model["ref_ff"], mass_ev)
                )
                rate_ref = compute_rate(
                    vdf_obj, ref_ff, model_name, model["ref_ff"], mass_ev
                )
                rate_universal = compute_rate(
                    vdf_obj, universal_ff, model_name, model["test_ff"], mass_ev
                )
                results.append(
                    {
                        "material": material,
                        "model": model_name,
                        "mass_MeV": mass_ev / 1e6,
                        "q_max_ref": q_max_ref,
                        "q_cut": q_cut,
                        "rate_ref": rate_ref,
                        "rate_universal": rate_universal,
                    }
                )
                print(
                    f"  {material}/{model_name}/{mass_label(mass_ev)}: "
                    f"ref={rate_ref:.4e}  universal={rate_universal:.4e}",
                    flush=True,
                )

    # --- 4. Save ---
    csv_path = cmp_dir / "results.csv"
    with open(str(csv_path), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {len(results)} rows to {csv_path}")


if __name__ == "__main__":
    main()
