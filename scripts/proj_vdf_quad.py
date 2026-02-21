"""
Project the Standard Halo Model (SHM) VDF using Gauss-Legendre quadrature.

Unlike proj_vdf.py which uses a Monte Carlo grid, this script uses adaptive
Gauss-Legendre quadrature for higher numerical precision.  The result is
useful for validating the Monte Carlo projection or generating reference data.

Usage:
    python proj_vdf_quad.py
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
    n0 = (                           # SHM normalisation constant
        np.pi ** (3 / 2)
        * v_0**2
        * (
            v_0 * special.erf(v_esc / v_0)
            - 2 * v_esc / np.sqrt(np.pi) * np.exp(-(v_esc**2) / v_0**2)
        )
    )

    physics_params = {
        "vdf": vdf_shm,
        "vdf_params": {"v_0": v_0, "v_e": v_e, "v_esc": v_esc, "n0": n0},
        "model": "SHM",              # label used as the HDF5 group prefix
    }

    # Use high precision parameters for validation
    numerics_params = {
        "v_max": (const.VESC + const.VE) * 1.0,  # velocity integration upper limit (eV)
        "l_max": 8,                  # max angular momentum quantum number
        "n_max": 127,                # 2**7 - 1, standard high resolution
    }

    hdf5_group = f'{physics_params["model"]}/230_240_600'

    # --- Quad-based projection ---
    print("=" * 60)
    print("Quad-based projection")
    print("=" * 60)

    quad_params = {
        "n_gl": 25,     # Gauss-Legendre nodes for cos(theta)
        "n_phi": 25,    # uniform nodes for phi
        "epsabs": 1e-8, # absolute tolerance for the radial integral
        "epsrel": 1e-8, # relative tolerance for the radial integral
        "limit": 200,   # max number of adaptive subdivisions
    }

    vdf_quad = VDF(physics_params=physics_params, numerics_params=numerics_params)
    vdf_quad.project_quad(params=quad_params, verbose=True)
    vdf_quad.export_hdf5(
        filename=str(output_dir / "vdf_quad.hdf5"),
        groupname=hdf5_group,
        dataname="data",
    )


if __name__ == "__main__":
    main()
