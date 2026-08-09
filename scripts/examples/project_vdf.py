"""
Project the Standard Halo Model (SHM) velocity distribution function (VDF)
onto the wavelet-harmonic basis by numerical integration on an
(n_r, n_theta, n_phi) grid.

Usage:
    python scripts/examples/project_vdf.py
"""

import logging
from pathlib import Path

import numba
import numpy as np
from scipy import special

from vectorphonodark import constants as const
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
    # Show the library's progress logs.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- SHM parameters ---
    # The VDF is projected once, in the frame with +z along the DM wind axis
    # at t = 0, so the Earth's velocity there is exactly (0, 0, v_E). Daily
    # modulation enters later by rotating the crystal (see Rotation).
    # most probable DM speed in the galactic frame (units of c)
    v_0 = const.V0
    # Earth's velocity at t = 0 (units of c)
    v_e = np.array([0.0, 0.0, const.VE])
    # galactic escape speed (units of c)
    v_esc = const.VESC
    # SHM normalization constant
    n0 = (
        np.pi ** (3 / 2)
        * v_0**2
        * (
            v_0 * special.erf(v_esc / v_0)
            - 2 * v_esc / np.sqrt(np.pi) * np.exp(-(v_esc**2) / v_0**2)
        )
    )

    physics_params = {
        # vdf to be projected
        "vdf": vdf_shm,
        "vdf_params": {"v_0": v_0, "v_e": v_e, "v_esc": v_esc, "n0": n0},
        # label used as the HDF5 group prefix
        "model": "SHM",
    }
    # (n_r, n_theta, n_phi) integration grid
    n_grid = (128, 180, 180)
    numerics_params = {
        # velocity integration upper limit (units of c)
        "v_max": (const.VESC + const.VE) * 1.0,
        # max multipole
        "l_max": 5,
        # max radial wavelet index
        "n_max": 2**7 - 1,
        "n_grid": n_grid,
    }
    file_params = {
        "hdf5": str(output_dir / "vdf.hdf5"),
        # One group per grid, so a re-run at another resolution does not
        # overwrite this one.
        "hdf5_group": "{}/grid_{}x{}x{}".format(physics_params["model"], *n_grid),
        "hdf5_data": "data",
    }

    params = {**physics_params, **numerics_params}

    vdf = VDF(physics_params=physics_params, numerics_params=numerics_params)
    vdf.project(params=params)
    vdf.export_hdf5(
        filename=file_params["hdf5"],
        groupname=file_params["hdf5_group"],
        dataname=file_params["hdf5_data"],
    )


if __name__ == "__main__":
    main()
