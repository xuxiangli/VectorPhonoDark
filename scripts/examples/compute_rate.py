"""
Compute the DM-phonon scattering rate and projected experimental reach.

Reads pre-computed VDF and form factor projections from HDF5 files, contracts
them for each DM mass, and applies the Wigner-G coefficients for the crystal
rotations at the chosen times (time_list; a single time by default). The result
is converted to a projected cross-section sensitivity and printed.

Usage:
    python scripts/examples/compute_rate.py
"""

import logging
from pathlib import Path

import numpy as np

from vectorphonodark import Rotation
from vectorphonodark import constants as const
from vectorphonodark.projection import VDF, FormFactor
from vectorphonodark.rate import Rate

logger = logging.getLogger(__name__)


def main():
    # Show the library's progress logs.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- DM model ---
    # DM masses (eV)
    mass_list = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]) * 10**6
    # mediator form factor power: F_DM ~ (q0/q)^f_med
    # 2 for light mediator and 0 for heavy mediator
    f_med = 2
    # for hadrophilic
    # reference momentum transfer (eV)
    q0_fdm = mass_list * const.V0
    # SM target particle mass (eV)
    mass_sm = const.M_NUCL
    # for dark photon
    # q0_fdm = [const.ALPHA_EM * const.M_ELEC] * len(mass_list)
    # mass_sm = const.M_ELEC

    # --- Numerics ---
    # # max radial wavelet index for v (optional)
    # nv_max = 2**7 - 1
    # # max radial wavelet index for q (optional)
    # nq_max = 2**9 - 1
    # # max multipole (optional)
    # l_max = 5

    # --- Rotation ---
    # list of times (hours) to evaluate the rate
    time_list = [0]
    # Crystal rotation for each time step. Active rotation acting on crystal;
    # see the Rotation class for the convention.
    _axis = np.array([0.0, -np.sin(const.THETA_E), np.cos(const.THETA_E)])
    rotations = [
        Rotation.from_axis_angle(_axis, 2.0 * np.pi * t / 24.0) for t in time_list
    ]

    # --- Reach calculation ---
    # signal threshold for reach projection
    events_per_year = 3.0
    # DM energy density prefactor in the rate
    factor = const.RHO_DM

    # --- Input files ---
    file_params_vdf = {
        "hdf5": str(output_dir / "vdf.hdf5"),
        "hdf5_group": "SHM/grid_128x180x180",
        "hdf5_data": "data",
    }
    file_params_form_factor = {
        "hdf5": str(output_dir / "GaAs_hadrophilic.hdf5"),
        "hdf5_group": "q_cut",
        "hdf5_data": "data",
    }

    # Load the VDF projection
    vdf = VDF().import_hdf5(
        filename=file_params_vdf["hdf5"],
        groupname=file_params_vdf["hdf5_group"],
        dataname=file_params_vdf["hdf5_data"],
    )

    # Load the form factor projection
    form_factor = FormFactor().import_hdf5(
        filename=file_params_form_factor["hdf5"],
        groupname=file_params_form_factor["hdf5_group"],
        dataname=file_params_form_factor["hdf5_data"],
    )

    for mass, q0 in zip(mass_list, q0_fdm):
        physics_params = {
            # rate ~ F_DM^2 = (q0/q)^(2*f_med): (a, b) in (q/q0)^a (v/v0)^b
            "fdm": (-2 * f_med, 0),
            # reference momentum for mediator form factor (eV)
            "q0_fdm": q0,
            # DM mass (eV)
            "mass_dm": mass,
            # SM target mass (eV)
            "mass_sm": mass_sm,
        }
        numerics_params = {
            # "l_max": l_max,
            # "nv_max": nv_max,
            # "nq_max": nq_max,
        }

        rate = Rate(
            physics_params=physics_params,
            numerics_params=numerics_params,
            vdf=vdf,
            ff=form_factor,
        )

        # Sum over energy bins to get the total rate per rotation
        rate_r = sum(rate.binned_rate(rotations).values())

        for i_rot in range(len(rotations)):
            reach = (
                events_per_year
                / const.KG_YR
                / (factor * float(rate_r[i_rot]))
                * const.INVEV_TO_CM**2
            )
            print(f"Mass {mass / 10**6} MeV, f_med {f_med}, rotation {i_rot}:")
            print(
                f"    Projected reach for {events_per_year} events "
                f"per year: {float(reach):.4e} cm^2"
            )


if __name__ == "__main__":
    main()
