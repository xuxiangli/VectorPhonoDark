import numpy as np
import numba
from scipy import special

from vectorphonodark import constants as const
from vectorphonodark import physics
from vectorphonodark import utility
from vectorphonodark import projection


"""Inputs start here"""
output_path = '/Users/jukcoeng/Desktop/Dark_Matter/Vector Space Integration/VectorPhonoDark/'

v_max = (const.VESC + const.VE) * 1.


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


t         = 0.0
v_0       = const.V0
v_e       = physics.create_vE_vec(t)
v_esc     = const.VESC
n0_factor = np.pi**(3/2) * v_0**2 * (
            v_0 * special.erf(v_esc / v_0) 
            - 2 * v_esc / np.sqrt(np.pi) * np.exp(-v_esc**2 / v_0**2)
        )

n_max = 2**7 - 1                # maximal radial basis index
l_max = 8                       # maximal angular basis index
physics_params = {
    'v_max': v_max,             # maximal velocity in eV
    'vdf_params': (v_0, v_e, v_esc, n0_factor) # parameters for vdf function, in order
    }
numerics_params = {
    'n_r':          128,        # number of r grid points
    'n_theta':      180,        # number of theta grid points
    'n_phi':        180,        # number of phi grid points
    'power_r':      1,          # power for r grid spacing
    'power_theta':  1,          # power for theta grid spacing
    'power_phi':    1,          # power for phi grid spacing
    'basis': 'haar',            # basis type
    }
file_params = {
    'vdf_model': 'shm',
    'csvname': output_path+'output/vdf/shm_230_240_600_128_180_180'
    }
"""Inputs end here"""


projection.proj_vdf(n_max=n_max, l_max=l_max, vdf=vdf_shm,
                    physics_params=physics_params, 
                    numerics_params=numerics_params,
                    file_params=file_params,
                    verbose=True)