import numpy as np
import numba
from scipy import special

from vectorphonodark import constants as const
from vectorphonodark import physics
from vectorphonodark import projection


"""Inputs start here"""
output_path = '/Users/jukcoeng/Desktop/Dark_Matter/Vector Space Integration/VectorPhonoDark/'


@numba.njit
def vdf_shm(v_xyz, v_0, v_e, v_esc, n0_factor) -> float:
    """
        Standard Halo Model velocity distribution function
    """

    v_gal_frame = np.linalg.norm(v_xyz + v_e)
    if v_gal_frame <= v_esc:
        vdf = np.exp(-v_gal_frame**2 / v_0**2) / n0_factor
    else:
        vdf = 0.0

    return vdf


t = 0.0
v_0 = const.V0
v_e = physics.create_vE_vec(t)
v_esc = const.VESC
n0_factor = np.pi**(3/2) * v_0**2 * (
    v_0 * special.erf(v_esc / v_0)
    - 2 * v_esc / np.sqrt(np.pi) * np.exp(-v_esc**2 / v_0**2)
)

l_list = list(range(0, 9))
n_list = list(range(2**7))

v_max = (const.VESC + const.VE) * 1.

physics_params = {
    'v_max': v_max,             # maximal velocity in eV
    'vdf': vdf_shm,
    'vdf_params': (v_0, v_e, v_esc, n0_factor)
}
numerics_params = {
    'l_list':               l_list,
    'n_list':               n_list,
    'n_a':                  128,         # number of r grid points
    'n_b':                  180,         # number of theta grid points
    'n_c':                  180,         # number of phi grid points
    'power_a':              1,           # power for r grid spacing
    # 'power_b':            1,           # power for theta grid spacing
    # 'power_c':            1,           # power for phi grid spacing
    # 'basis':                'haar',    # basis type
}
file_params = {
    'vdf_model': 'shm',
    'csvname': output_path+'output/vdf/shm_230_240_600_128_180_180_new'
}
"""Inputs end here"""


projection.proj_vdf(physics_params=physics_params,
                    numerics_params=numerics_params,
                    file_params=file_params,
                    verbose=True)
