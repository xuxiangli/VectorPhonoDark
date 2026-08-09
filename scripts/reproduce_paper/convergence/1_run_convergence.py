"""Run the convergence study behind the draft's appendix tables.

One-parameter scans of the rate around the default configuration
(l_max = 8, N_v = 128, N_q = 512, VDF grid (128, 180, 180), FF grid
(512, 25, 25), log wavelets, 1 meV bins), for 2 materials x 3 DM models
x 4 masses:

  scan       values                reference
  l_max      2, 5                  8
  n_vmax     63, 127               255      (N_v = 64, 128 vs 256)
  n_qmax     511, 1023             2047     (N_q = 512, 1024 vs 2048, on the
                                            FF grid (2048, 25, 25))
  vdf_n_r    128, 256              512
  vdf_n_ang  90, 180               360
  ff_n_r     512, 1024             2048
  ff_n_ang   15, 25                50

Projections are shared wherever truncation allows: the l_max = 8
projections serve every l_max <= 8 row, the nv_max = 255 VDF serves every
smaller N_v, the nq_max = 2047 form factor serves every N_q row, and the
heavy/light hadrophilic models share material form factors.
Projections and rates are cached under output/convergence/, so an
interrupted run resumes. 2_make_tables.py turns results.csv into the seven
LaTeX tables (tab:conv2_*).

Usage:
    python scripts/reproduce_paper/convergence/1_run_convergence.py
"""

import os

# Pin BLAS to one thread before numpy loads: the form-factor projection is
# dominated by tiny per-q-point eigensolves, which multithreaded BLAS makes
# slower. An explicit setting in the environment still wins over this default.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import csv
import json
import time
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
MASSES_EV = [0.1e6, 1.0e6, 10.0e6, 100.0e6]

MODELS = {
    "heavy_hadrophilic": {
        "fdm": (0, 0),
        "mass_sm": const.M_NUCL,
        "q0_fdm_func": lambda m: m * const.V0,
        "ff_model": "hadrophilic",
    },
    "light_hadrophilic": {
        "fdm": (-4, 0),
        "mass_sm": const.M_NUCL,
        "q0_fdm_func": lambda m: m * const.V0,
        "ff_model": "hadrophilic",
    },
    "light_dark_photon": {
        "fdm": (-4, 0),
        "mass_sm": const.M_ELEC,
        "q0_fdm_func": lambda _m: const.ALPHA_EM * const.M_ELEC,
        "ff_model": "dark_photon",
    },
}

L_BASE = 8
NV_BASE, NV_REF = 127, 255
NQ_BASE = 511

# VDF projections: label -> (l_max, nv_max, n_grid). "base" serves the l_max
# scan (truncated at rate time) and every base row; "nv255" serves the N_v
# scan and, truncated to nv_max = 127, the 256-point radial-grid row.
VDF_SPECS = {
    "base": (L_BASE, NV_BASE, (128, 180, 180)),
    "nv255": (L_BASE, NV_REF, (256, 180, 180)),
    "nr512": (L_BASE, NV_BASE, (512, 180, 180)),
    "nang90": (L_BASE, NV_BASE, (128, 90, 90)),
    "nang360": (L_BASE, NV_BASE, (128, 360, 360)),
}

# Form-factor projections per (material, coupling, mass): label -> (l_max,
# n_grid, nq_max). All on the mass's own kinematic domain. "nq2047" carries
# the full N_q = 2048 radial basis, which requires the 2048-point radial
# grid (the package needs N_r >= N_q).
FF_SPECS = {
    "base": (L_BASE, (512, 25, 25), NQ_BASE),
    "nr1024": (L_BASE, (1024, 25, 25), NQ_BASE),
    "nr2048": (L_BASE, (2048, 25, 25), NQ_BASE),
    "nang15": (L_BASE, (512, 15, 15), NQ_BASE),
    "nang50": (L_BASE, (512, 50, 50), NQ_BASE),
    "nq2047": (L_BASE, (2048, 25, 25), 2047),
}

# Scans: value -> (vdf_label, nv_max_rate, ff_label, l_max_rate,
# nq_max_rate). The base configuration recurs across scans; the rate cache
# dedups it. The n_qmax rows all truncate the single nq2047 projection, so
# the scan isolates the radial-basis truncation from grid effects.
SCANS = {
    "l_max": {ell: ("base", NV_BASE, "base", ell, NQ_BASE) for ell in (2, 5, 8)},
    "n_vmax": {nv: ("nv255", nv, "base", L_BASE, NQ_BASE) for nv in (63, 127, 255)},
    "n_qmax": {nq: ("base", NV_BASE, "nq2047", L_BASE, nq) for nq in (511, 1023, 2047)},
    "vdf_n_r": {
        128: ("base", NV_BASE, "base", L_BASE, NQ_BASE),
        256: ("nv255", NV_BASE, "base", L_BASE, NQ_BASE),
        512: ("nr512", NV_BASE, "base", L_BASE, NQ_BASE),
    },
    "vdf_n_ang": {
        90: ("nang90", NV_BASE, "base", L_BASE, NQ_BASE),
        180: ("base", NV_BASE, "base", L_BASE, NQ_BASE),
        360: ("nang360", NV_BASE, "base", L_BASE, NQ_BASE),
    },
    "ff_n_r": {
        512: ("base", NV_BASE, "base", L_BASE, NQ_BASE),
        1024: ("base", NV_BASE, "nr1024", L_BASE, NQ_BASE),
        2048: ("base", NV_BASE, "nr2048", L_BASE, NQ_BASE),
    },
    "ff_n_ang": {
        15: ("base", NV_BASE, "nang15", L_BASE, NQ_BASE),
        25: ("base", NV_BASE, "base", L_BASE, NQ_BASE),
        50: ("base", NV_BASE, "nang50", L_BASE, NQ_BASE),
    },
}

ENERGY_THRESHOLD = 1e-3
ENERGY_BIN_WIDTH = 1e-3

VDF_GROUP = "SHM/t0"
FF_GROUP = "data"
HDF5_DATA = "data"

CSV_FIELDS = ["scan", "material", "model", "mass_MeV", "value", "rate"]

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


# =====================  Paths  =====================


def material_input(project_root, material):
    return str(project_root / "inputs" / "material" / material / f"{material}.py")


def vdf_path(conv_dir, label):
    return conv_dir / "vdf" / f"{label}.hdf5"


def ff_path(conv_dir, material, ff_model, mass_ev, label):
    return (
        conv_dir
        / "ff"
        / material
        / ff_model
        / f"{mass_ev / 1e6:g}MeV"
        / (label + ".hdf5")
    )


def q_max_for_mass(mass_ev, q_cutoff):
    return min(2.0 * mass_ev * (const.VESC + const.VE), q_cutoff)


# =====================  Projections  =====================


def ensure_vdf(conv_dir, label):
    path = vdf_path(conv_dir, label)
    if path.exists():
        return
    l_max, nv_max, n_grid = VDF_SPECS[label]
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  VDF {label}: l_max={l_max}, nv_max={nv_max}, n_grid={n_grid}")
    t0 = time.time()
    shm = _shm_params()
    pp = {"vdf": vdf_shm, "vdf_params": shm, "model": "SHM"}
    np_ = {
        "v_max": const.VESC + const.VE,
        "l_max": l_max,
        "n_max": nv_max,
        "n_grid": n_grid,
    }
    vdf = VDF(physics_params=pp, numerics_params=np_)
    vdf.project(params={**pp, **np_})
    vdf.export_hdf5(filename=str(path), groupname=VDF_GROUP, dataname=HDF5_DATA)
    print(f"    saved -> {path.name}  ({time.time() - t0:.1f} s)")


def ensure_ff(project_root, conv_dir, material, ff_model, mass_ev, label, q_cutoff):
    path = ff_path(conv_dir, material, ff_model, mass_ev, label)
    if path.exists():
        return
    l_max, n_grid, nq_max = FF_SPECS[label]
    q_max = q_max_for_mass(mass_ev, q_cutoff)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"  FF {material}/{ff_model}/{mass_ev / 1e6:g}MeV {label}: "
        f"l_max={l_max}, nq_max={nq_max}, n_grid={n_grid}, q_max={q_max:.4e}",
        flush=True,
    )
    t0 = time.time()
    pp = {
        "energy_threshold": ENERGY_THRESHOLD,
        "energy_bin_width": ENERGY_BIN_WIDTH,
        "energy_max_factor": 1.2,
        "model": f"{material}_{ff_model}",
    }
    np_ = {
        "q_max": q_max,
        "l_max": l_max,
        "n_max": nq_max,
        "n_grid": n_grid,
        "log_wavelet": True,
    }
    ip = {
        "material_input": material_input(project_root, material),
        "physics_model_input": str(
            project_root / "inputs" / "physics_model" / f"{ff_model}.py"
        ),
        "numerics_input": str(project_root / "inputs" / "numerics" / "standard.py"),
    }
    ff = FormFactor(physics_params=pp, numerics_params=np_)
    ff.project(params={**pp, **np_, **ip})
    ff.export_hdf5(filename=str(path), groupname=FF_GROUP, dataname=HDF5_DATA)
    print(f"    saved -> {path.name}  ({(time.time() - t0) / 60:.1f} min)")


# =====================  Rates  =====================


def compute_rate(vdf, ff, model_name, mass_ev, l_max_rate, nv_max_rate, nq_max_rate):
    cfg = MODELS[model_name]
    pp = {
        "fdm": cfg["fdm"],
        "q0_fdm": cfg["q0_fdm_func"](mass_ev),
        "mass_dm": mass_ev,
        "mass_sm": cfg["mass_sm"],
    }
    np_ = {"l_max": l_max_rate, "nv_max": nv_max_rate, "nq_max": nq_max_rate}
    rate = Rate(physics_params=pp, numerics_params=np_, vdf=vdf, ff=ff)
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
    conv_dir = project_root / "output" / "convergence"
    conv_dir.mkdir(parents=True, exist_ok=True)

    print("Warming up Numba JIT ...")
    _warmup()

    q_cutoff = {
        material: phonopy_funcs.compute_q_cut(material_input(project_root, material))
        for material in MATERIALS
    }
    for material, q_cut in q_cutoff.items():
        print(f"  q_cutoff[{material}] = {q_cut:.6e} eV")

    # --- 1. Projections ---
    print("\n=== Projecting VDFs ===")
    for label in VDF_SPECS:
        ensure_vdf(conv_dir, label)

    print("\n=== Projecting material form factors ===")
    ff_models = sorted({cfg["ff_model"] for cfg in MODELS.values()})
    for material in MATERIALS:
        for ff_model in ff_models:
            for mass_ev in MASSES_EV:
                for label in FF_SPECS:
                    ensure_ff(
                        project_root,
                        conv_dir,
                        material,
                        ff_model,
                        mass_ev,
                        label,
                        q_cutoff[material],
                    )

    # --- 2. Rates ---
    # Cached by the full rate configuration, so the base configuration that
    # recurs in four scans is computed once per (material, model, mass).
    cache_path = conv_dir / "rate_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    # Keys predating the N_q scan lack the |nq field; they were all computed
    # at nq_max = NQ_BASE.
    cache = {(k if "|nq" in k else f"{k}|nq{NQ_BASE}"): v for k, v in cache.items()}

    vdfs = {
        label: VDF().import_hdf5(
            filename=str(vdf_path(conv_dir, label)),
            groupname=VDF_GROUP,
            dataname=HDF5_DATA,
        )
        for label in VDF_SPECS
    }

    print("\n=== Computing rates ===")
    results = []
    for material in MATERIALS:
        for model_name, cfg in MODELS.items():
            for mass_ev in MASSES_EV:
                ffs = {}  # ff_label -> loaded FormFactor, per case

                for scan, values in SCANS.items():
                    for value, (
                        vdf_label,
                        nv_max,
                        ff_label,
                        l_max,
                        nq_max,
                    ) in values.items():
                        key = (
                            f"{material}|{model_name}|{mass_ev / 1e6:g}|"
                            f"{vdf_label}|{ff_label}|l{l_max}|nv{nv_max}|nq{nq_max}"
                        )
                        if key not in cache:
                            if ff_label not in ffs:
                                ffs[ff_label] = FormFactor().import_hdf5(
                                    filename=str(
                                        ff_path(
                                            conv_dir,
                                            material,
                                            cfg["ff_model"],
                                            mass_ev,
                                            ff_label,
                                        )
                                    ),
                                    groupname=FF_GROUP,
                                    dataname=HDF5_DATA,
                                )
                            t0 = time.time()
                            cache[key] = compute_rate(
                                vdfs[vdf_label],
                                ffs[ff_label],
                                model_name,
                                mass_ev,
                                l_max,
                                nv_max,
                                nq_max,
                            )
                            cache_path.write_text(json.dumps(cache, indent=2))
                            print(
                                f"  {key}: {cache[key]:.6e}  "
                                f"({time.time() - t0:.1f} s)",
                                flush=True,
                            )
                        results.append(
                            {
                                "scan": scan,
                                "material": material,
                                "model": model_name,
                                "mass_MeV": mass_ev / 1e6,
                                "value": value,
                                "rate": cache[key],
                            }
                        )

    # --- 3. Save ---
    csv_path = conv_dir / "results.csv"
    with open(str(csv_path), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {len(results)} rows to {csv_path}")
    print("Next: 2_make_tables.py")


if __name__ == "__main__":
    main()
