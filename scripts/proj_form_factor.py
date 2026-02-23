"""
Project the dark matter form factor onto the spherical wavelet basis.

For each DM mass in mass_list, the kinematically accessible momentum range
[0, q_max] is computed and the crystal form factor is projected.  
Results are written to an HDF5 file, one group per q_max value.

Usage:
    python proj_form_factor.py
"""

import numpy as np
from pathlib import Path

from vectorphonodark import constants as const
from vectorphonodark.projection import FormFactor


def main():
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # q_max_list = [2.5*10**2, 1.0*10**3, 2.5*10**3, 1.0*10**4, 2.5*10**4, 1.0*10**5, 2.5*10**5]

    mass_list = np.array([0.01, 0.03]) * 10**6          # DM masses in eV
    q_max_list = [2 * mass * (const.VESC + const.VE) for mass in mass_list]  # kinematic upper limit per mass
    # q_cutoff = 470672.7665347568  # for GaAs
    # q_cutoff = 473779.35795407614  # for Al2O3
    # q_max_list = [min(q_max, q_cutoff) for q_max in q_max_list]

    for q_max in q_max_list:
        physics_params = {
            "energy_threshold": 1e-3,   # eV; minimum detectable phonon energy
            "energy_bin_width": 1e-3,   # eV; energy bin size for the spectrum
            # factor to multiply with Gamma point to get energy cutoff
            # default: 4.0
            "energy_max_factor": 1.2,
            "model": "Al2O3_dark_photon",  # output label
        }
        numerics_params = {
            "q_max": q_max,         # upper momentum cutoff in eV
            "l_max": 8,             # max angular momentum quantum number
            "n_max": 2**9 - 1,      # max radial wavelet index
            "n_grid": (2**9, 25, 25),  # (n_r, n_theta, n_phi) quadrature grid
            "log_wavelet": True,    # use logarithmic radial wavelet basis, True for light DM and False for heavy DM
        }
        input_params = {
            "material_input": str(project_root / "inputs" / "material" / "Al2O3" / "Al2O3.py"),
            "physics_model_input": str(project_root / "inputs" / "physics_model" / "dark_photon.py"),
            "numerics_input": str(project_root / "inputs" / "numerics" / "standard.py"),
        }
        file_params = {
            "hdf5": str(output_dir / f"{physics_params['model']}.hdf5"),
            "hdf5_group": f"log/{q_max} eV",  # group name encodes wavelet type and q_max
            "hdf5_data": "data",
        }

        params = {**physics_params, **numerics_params, **input_params}

        form_factor = FormFactor(
            physics_params=physics_params, numerics_params=numerics_params
        )
        form_factor.project(params=params, verbose=True)
        form_factor.export_hdf5(
            filename=file_params["hdf5"],
            groupname=file_params["hdf5_group"],
            dataname=file_params["hdf5_data"],
        )


if __name__ == "__main__":
    main()
