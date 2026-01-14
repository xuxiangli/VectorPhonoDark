import numpy as np
import numba
from scipy import special

from vectorphonodark import constants as const
from vectorphonodark.projection import VDF
from vectorphonodark import physics


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
n0 = (
    np.pi ** (3 / 2)
    * v_0**2
    * (
        v_0 * special.erf(v_esc / v_0)
        - 2 * v_esc / np.sqrt(np.pi) * np.exp(-(v_esc**2) / v_0**2)
    )
)

output_path = "/Users/jukcoeng/Desktop/Dark_Matter/Vector Space Integration/VectorPhonoDark/output/"

physics_params = {
    "vdf": vdf_shm,
    "vdf_params": (v_0, v_e, v_esc, n0),
    "vdf_params": {"v_0": v_0, "v_e": v_e, "v_esc": v_esc, "n0": n0},
    "model": "SHM",
}
numerics_params = {
    # reference velocity scale
    "v_max": (const.VESC + const.VE) * 1.0,
    "l_max": 5,
    "n_list": list(range(2**7)),
    "n_grid": (128, 180, 180),
    # 'basis': 'haar',
}
file_params = {
    "csv": output_path
    + f"vdf/{physics_params['model']}_230_240_600_{numerics_params['n_grid']}"
    + ".csv",
    "hdf5": output_path + "VDF" + ".hdf5",
    "hdf5_group": f"vdf/{physics_params['model']}/230_240_600/{numerics_params['n_grid']}",
    "hdf5_data": "data",
}
params = {**physics_params, **numerics_params}

vdf = VDF(physics_params=physics_params, numerics_params=numerics_params)
vdf.project(params=params, verbose=True)
# vdf.export_csv(filename=file_params['csv'])
vdf.export_hdf5(
    filename=file_params["hdf5"],
    groupname=file_params["hdf5_group"],
    dataname=file_params["hdf5_data"],
)
