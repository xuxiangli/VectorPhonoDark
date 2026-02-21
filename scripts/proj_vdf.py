"""
Project the Standard Halo Model (SHM) velocity distribution function (VDF)
onto the spherical wavelet basis via Monte Carlo grid integration.

The VDF f(v) gives the probability density of DM velocities in the lab frame.
Its spherical wavelet coefficients are required as input for the rate
calculation in rate_calc.py.

Usage:
    python proj_vdf.py
"""

import numpy as np
import numba
from scipy import special
from pathlib import Path

from vectorphonodark import constants as const
from vectorphonodark import physics
from vectorphonodark.projection import VDF


@numba.njit
def vdf_shm(v_xyz, v_0, v_e, v_esc, n0) -> float:
    """
    Standard Halo Model velocity distribution function
    """

    v_gal_frame = np.linalg.norm(v_xyz + v_e)
    if v_gal_frame <= v_esc:
        vdf = np.exp(-(v_gal_frame**2) / v_0**2) / n0
    else:
        vdf = 0.0

    return vdf


def main():
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- SHM parameters ---
    t = 0.0                          # time of the day (hours)
    v_0 = const.V0                   # most probable DM speed in galactic frame (eV)
    v_e = physics.create_vE_vec(t)   # Earth's velocity vector at time t (eV)
    v_esc = const.VESC               # galactic escape speed (eV)
    n0 = (                           # SHM normalization constant
        np.pi ** (3 / 2)
        * v_0**2
        * (
            v_0 * special.erf(v_esc / v_0)
            - 2 * v_esc / np.sqrt(np.pi) * np.exp(-(v_esc**2) / v_0**2)
        )
    )

    physics_params = {
        "vdf": vdf_shm,              # vdf to be projected
        "vdf_params": {"v_0": v_0, "v_e": v_e, "v_esc": v_esc, "n0": n0},
        "model": "SHM",              # label used as the HDF5 group prefix
    }
    numerics_params = {
        "v_max": (const.VESC + const.VE) * 1.0,  # velocity integration upper limit (eV)
        "l_max": 15,                 # max angular momentum quantum number
        "n_max": 2**7 - 1,           # max radial wavelet index
        "n_grid": (512, 180, 180),   # (n_r, n_theta, n_phi) Monte Carlo grid
    }
    file_params = {
        "hdf5": str(output_dir / "vdf_2.hdf5"),
        "hdf5_group": f'{physics_params["model"]}/230_240_600',  # encodes grid size
        "hdf5_data": "data",
    }

    params = {**physics_params, **numerics_params}

    vdf = VDF(physics_params=physics_params, numerics_params=numerics_params)
    vdf.project(params=params, verbose=True)
    vdf.export_hdf5(
        filename=file_params["hdf5"],
        groupname=file_params["hdf5_group"],
        dataname=file_params["hdf5_data"],
    )


if __name__ == "__main__":
    main()
