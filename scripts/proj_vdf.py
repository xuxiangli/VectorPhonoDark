import numpy as np
import numba
from scipy import special
import os
from pathlib import Path

from vectorphonodark import constants as const
from vectorphonodark import physics
from vectorphonodark.projection import VDF


"""input starts here"""
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


t = 0.0
v_0 = const.V0
v_e = physics.create_vE_vec(t)
v_esc = const.VESC
n0 = (np.pi ** (3 / 2) * v_0**2 * (
        v_0 * special.erf(v_esc / v_0)
        - 2 * v_esc / np.sqrt(np.pi) * np.exp(-(v_esc**2) / v_0**2)
    )
)

project_root = Path(__file__).resolve().parent.parent
output_path = str(project_root / "output") + "/"

if not os.path.exists(output_path):
    os.makedirs(output_path)

physics_params = {
    "vdf": vdf_shm,
    "vdf_params": (v_0, v_e, v_esc, n0),
    "vdf_params": {"v_0": v_0, "v_e": v_e, "v_esc": v_esc, "n0": n0},
    "model": "SHM",
}
numerics_params = {
    "v_max": (const.VESC + const.VE) * 1.0,
    "l_max": 5,
    "n_max": 2**7 - 1,
    "n_grid": (512, 180, 180),
}
file_params = {
    "hdf5": output_path + "vdf" + ".hdf5",
    "hdf5_group": f'{physics_params["model"]}/230_240_600',
    "hdf5_data": "data",
}
"""input ends here"""

params = {**physics_params, **numerics_params}

vdf = VDF(physics_params=physics_params, numerics_params=numerics_params)
vdf.project(params=params, verbose=True)
vdf.export_hdf5(
    filename=file_params["hdf5"],
    groupname=file_params["hdf5_group"],
    dataname=file_params["hdf5_data"],
)
