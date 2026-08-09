"""
Project the material form factor onto the wavelet-harmonic basis.

One universal projection per (material, coupling): GaAs and Al2O3, each
with the hadrophilic and dark photon couplings, projected onto the
logarithmic wavelet basis up to the Debye-Waller cutoff q_cut.

Usage:
    python scripts/examples/project_material_ff.py
"""

import os

# Pin BLAS to one thread before numpy loads: the form-factor projection is
# dominated by tiny per-q-point eigensolves, which multithreaded BLAS makes
# slower. An explicit setting in the environment still wins over this default.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging
from pathlib import Path

from vectorphonodark import constants as const
from vectorphonodark import phonopy_funcs
from vectorphonodark.projection import FormFactor

MATERIALS = ["GaAs", "Al2O3"]
COUPLINGS = ["hadrophilic", "dark_photon"]

# max multipole
L_MAX = 5
# max radial wavelet index (N_q = 512)
NQ_MAX = 511
# (n_r, n_theta, n_phi) integration grid
N_GRID = (2048, 25, 25)
# minimum detectable phonon energy (eV)
ENERGY_THRESHOLD = 1e-3
# energy bin size for the spectrum (eV)
ENERGY_BIN_WIDTH = 1e-3
# energy cutoff, as a multiple of the largest Gamma-point phonon energy
ENERGY_MAX_FACTOR = 1.2


def main():
    # Show the library's progress logs.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    project_root = Path(__file__).resolve().parents[2]
    out_dir = project_root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    for material in MATERIALS:
        material_input = str(
            project_root / "inputs" / "material" / material / f"{material}.py"
        )
        # The Debye-Waller cutoff
        q_cut = phonopy_funcs.compute_q_cut(material_input)

        for coupling in COUPLINGS:
            model = f"{material}_{coupling}"
            out_path = out_dir / f"{model}.hdf5"
            if out_path.exists():
                print(f"[skip] {out_path.name} already exists")
                continue

            physics_params = {
                # minimum detectable phonon energy (eV)
                "energy_threshold": ENERGY_THRESHOLD,
                # energy bin size for the spectrum (eV)
                "energy_bin_width": ENERGY_BIN_WIDTH,
                # Energy cutoff, as a multiple of the largest Gamma-point phonon
                # energy (default: 1.2).
                "energy_max_factor": ENERGY_MAX_FACTOR,
                # output label
                "model": model,
            }
            numerics_params = {
                # upper momentum cutoff (eV)
                "q_max": q_cut,
                # max multipole
                "l_max": L_MAX,
                # max radial wavelet index
                "n_max": NQ_MAX,
                # (n_r, n_theta, n_phi) integration grid
                "n_grid": N_GRID,
                # use logarithmic radial wavelet basis
                "log_wavelet": True,
                # minimum momentum transfer for logarithmic wavelet basis (eV)
                "q_min": ENERGY_THRESHOLD / (const.VESC + const.VE),
            }
            input_params = {
                "material_input": material_input,
                "physics_model_input": str(
                    project_root / "inputs" / "physics_model" / f"{coupling}.py"
                ),
                "numerics_input": str(
                    project_root / "inputs" / "numerics" / "standard.py"
                ),
            }
            file_params = {
                "hdf5": str(out_path),
                # group name in the HDF5 file
                "hdf5_group": "q_cut",
                # data name
                "hdf5_data": "data",
            }

            params = {**physics_params, **numerics_params, **input_params}

            form_factor = FormFactor(
                physics_params=physics_params, numerics_params=numerics_params
            )
            form_factor.project(params=params)
            form_factor.export_hdf5(
                filename=file_params["hdf5"],
                groupname=file_params["hdf5_group"],
                dataname=file_params["hdf5_data"],
            )


if __name__ == "__main__":
    main()
