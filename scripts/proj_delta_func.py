from vectorphonodark import constants as const
from vectorphonodark import projection


"""Inputs start here"""
input_path = '/Users/jukcoeng/Desktop/Dark_Matter/Vector Space Integration/VectorPhonoDark/'
output_path = '/Users/jukcoeng/Desktop/Dark_Matter/Vector Space Integration/VectorPhonoDark/'

mass_list = [10**8]
fdm_list = [2]

l_list = list(range(6))
nv_list = list(range(32))
nq_list = list(range(2**7)) + [2**p for p in range(7, 20)]

v_max = (const.VESC + const.VE) * 1.0
q_max = [2*mass*(const.VESC + const.VE) for mass in mass_list]

physics_params = {
    'v_max': v_max,
    'q_max': q_max,
    'threshold': 1e-3,           # eV
    'mass_list': mass_list,
    'fdm_list': fdm_list,
    'mass_sm': const.M_NUCL
}
numerics_params = {
    'l_list': l_list,
    'nv_list': nv_list,
    'nq_list': nq_list,
    'energy_bin_width': 1e-3,   # eV
    'energy_max': 36e-3, # 0.10723,      # eV, maximal energy for all materials considered
}
file_params = {
    'hdf5name': output_path+'output/test'
}
"""Inputs end here"""

projection.proj_mcalI(physics_params=physics_params, 
                      numerics_params=numerics_params, 
                      file_params=file_params, 
                      verbose=True)
