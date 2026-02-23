"""
Compute the DM-phonon scattering rate and projected experimental reach.

Reads pre-computed VDF and form factor projections from HDF5 files, contracts
them via the vsdm inner product for each DM mass, and applies Wigner-D
rotations to account for Earth's rotation.  The result is converted to a
projected cross-section sensitivity and printed.

Usage:
    python rate_calc.py
"""

import numpy as np
from pathlib import Path

import vsdm

from vectorphonodark import constants as const
from vectorphonodark import utility
from vectorphonodark.projection import VDF, FormFactor
from vectorphonodark.rate import Rate, import_all_form_factors


def main():
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- DM model ---
    mass_list = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]) * 10**6  # DM masses in eV
    f_med = 2                        # mediator form factor power: F_med ~ (q0/q)^f_med, 2 for light mediator and 0 for heavy mediator
    # for hadrophilic
    # q0_fdm = mass_list * const.V0  # reference momentum transfer in eV
    # mass_sm = const.M_NUCL
    # for dark photon
    q0_fdm = [const.ALPHA_EM * const.M_ELEC] * len(mass_list)  # reference momentum (eV)
    mass_sm = const.M_ELEC          # SM target particle mass (eV)

    # --- Numerics ---
    nv_max = 2**7 - 1               # max radial wavelet index for VDF
    nq_max = 2**8 - 1               # max radial wavelet index for form factor

    l_max = 5                       # max angular momentum quantum number
    time_list = [0]                  # list of times (hours) to evaluate the rate
    # rotation axis due to Earth's rotation
    # Axis used the same as in the vdf projection
    nx, ny, nz = 0, -np.sin(const.THETA_E), np.cos(const.THETA_E)

    # --- Reach calculation ---
    events_per_year = 3.0           # signal threshold for reach projection
    factor = const.RHO_DM           # additional factor in the rate calculation
    verbose = False

    # --- Input files ---
    file_params_vdf = {
        "hdf5": str(output_dir / "vdf.hdf5"),
        "hdf5_group": "SHM/230_240_600",
        "hdf5_data": "data",
        "verbose": True,
    }
    file_params_form_factor = {
        "hdf5": str(output_dir / "Al2O3_dark_photon.hdf5"),
        "hdf5_group": "log",
        "hdf5_data": "data",
        "verbose": True,
    }

    # Load the VDF projection
    vdf = VDF().import_hdf5(
        filename=file_params_vdf["hdf5"],
        groupname=file_params_vdf["hdf5_group"],
        dataname=file_params_vdf["hdf5_data"],
        verbose=file_params_vdf["verbose"],
    )

    # Load a single form factor for a specific q_max value
    # form_factor = FormFactor().import_hdf5(
    #     filename=file_params_form_factor["hdf5"],
    #     groupname=file_params_form_factor["hdf5_group"],
    #     dataname=file_params_form_factor["hdf5_data"],
    #     verbose=file_params_form_factor["verbose"],
    # )

    # Load all form factors indexed by their q_max value
    form_factors, q_max_list_file = import_all_form_factors(
        filename=file_params_form_factor["hdf5"],
        groupname=file_params_form_factor["hdf5_group"],
        dataname=file_params_form_factor["hdf5_data"],
        verbose=file_params_form_factor["verbose"],
    )

    # Build Wigner-D rotation matrices for each time step
    rotationlist = []
    conj_D = vsdm.WignerG(l_max).conj_D
    for t in time_list:
        phi = -2 * const.PI * t / 24.0   # rotation angle in radians
        q = utility.rot_to_quaternion(nx, ny, nz, phi)
        # if WignerG use conjugate D, we should conjugate the rotation to match the convention of Wigner D
        if conj_D:
            rotationlist.append(q.conjugate())
        else:
            rotationlist.append(q)
    wG = vsdm.WignerG(l_max, rotations=rotationlist)

    for mass, q0 in zip(mass_list, q0_fdm):
        q_max = 2 * mass * (const.VESC + const.VE)  # kinematic q_max for this mass

        physics_params = {
            "fdm": (-2 * f_med, 0),  # (a, b) with F_med = (q0/q)^a * (v0/v)^b
            "q0_fdm": q0,            # reference momentum for mediator form factor (eV)
            "mass_dm": mass,         # DM mass (eV)
            "mass_sm": mass_sm,      # SM target mass (eV)
        }
        numerics_params = {
            "l_max": l_max,
            "nv_max": nv_max,
            "nq_max": nq_max,
        }

        # Select the smallest stored q_max that covers the kinematic range
        q_max_file = q_max_list_file[q_max_list_file >= q_max]
        if len(q_max_file) == 0:
            print(
                "\n    Using form factor with largest q_max = "
                f"{q_max_list_file[-1]:.2e} eV for mass {mass / 10**6:.2f} MeV."
            )
            q_max_file = q_max_list_file[-1]
        else:
            print(
                "\n    Using form factor with q_max = "
                f"{q_max_file[0]:.2e} eV for mass {mass / 10**6:.2f} MeV "
                f"with optimized q_max = {q_max:.2e} eV."
            )
            q_max_file = q_max_file[0]

        form_factor = form_factors[q_max_file]

        binned_rate = Rate(
            physics_params=physics_params,
            numerics_params=numerics_params,
            vdf=vdf,
            ff=form_factor,
            verbose=verbose,
        )

        # Sum over energy bins to get the total rate per rotation
        rate_r = sum(binned_mu_R for binned_mu_R in binned_rate.binned_mu_R(wG=wG).values())

        for i_rot in range(len(rotationlist)):
            reach = (
                events_per_year
                / const.KG_YR
                / (factor * float(rate_r[i_rot]))
                * const.inveV_to_cm**2
            )
            print(f"Mass {mass / 10**6} MeV, f_med {f_med}, rotation {i_rot}:")
            print(
                f"    Projected reach for {events_per_year} events "
                f"per year: {float(reach):.4e} cm^2"
            )


if __name__ == "__main__":
    main()
