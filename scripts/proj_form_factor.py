import os

import phonopy

from vectorphonodark import constants as const
from vectorphonodark import phonopy_funcs
from vectorphonodark import utility
from vectorphonodark import projection


"""Inputs start here"""
input_path = '/Users/jukcoeng/Desktop/Dark_Matter/Vector Space Integration/VectorPhonoDark/'
output_path = '/Users/jukcoeng/Desktop/Dark_Matter/Vector Space Integration/VectorPhonoDark/'

l_list = list(range(6))
n_list = list(range(2**7))

mass = 10**7  # in eV
q_max = 2*mass*(const.VESC + const.VE)  # maximal momentum transfer in eV

physics_params = {
    'q_max': q_max,
    'threshold': 1e-3           # in eV
}
numerics_params = {
    'l_list':               l_list,
    'n_list':               n_list,
    'n_a':                  128,         # number of r grid points
    'n_b':                  25,         # number of theta grid points
    'n_c':                  25,         # number of phi grid points
    'power_a':              1,          # power for r grid spacing
    # 'power_b':            1,          # power for theta grid spacing
    # 'power_c':            1,          # power for phi grid spacing
    'special_mesh':         True,        # whether to use special mesh and wavelets
    'energy_bin_width':     1e-3,   # eV
    # factor to multiply with Gamma point to get energy cutoff
    # default is 4.0 if not specified
    'energy_max_factor':    1.2,
    # 'basis':                'haar',            # basis type
    'q_cut':                False               # whether to compute q_cut from Debye-Waller factor
}
file_params = {
    'modelname': 'GaAs',
    'csvname': output_path+'output/form_factor/GaAs/test/GaAs_hadrophilic_10MeV_128_25_25'
}

material_input = input_path+'inputs/material/GaAs/GaAs.py'
physics_model_input = input_path+'inputs/physics_model/hadrophilic.py'
numerics_input = input_path+'inputs/numerics/standard.py'
"""Inputs end here"""


mat_input_mod_name = os.path.splitext(os.path.basename(material_input))[0]
phys_input_mod_name = os.path.splitext(
    os.path.basename(physics_model_input))[0]
num_input_mod_name = os.path.splitext(os.path.basename(numerics_input))[0]

mat_mod = utility.import_file(mat_input_mod_name,
                              os.path.join(material_input))
phys_mod = utility.import_file(phys_input_mod_name,
                               os.path.join(physics_model_input))
num_mod = utility.import_file(num_input_mod_name,
                              os.path.join(numerics_input))

material = mat_mod.material
c_dict = phys_mod.c_dict

poscar_path = os.path.join(
    os.path.split(material_input)[0], 'POSCAR'
)
force_sets_path = os.path.join(
    os.path.split(material_input)[0], 'FORCE_SETS'
)
born_path = os.path.join(
    os.path.split(material_input)[0], 'BORN'
)

if os.path.exists(born_path):
    born_exists = True
else:
    print('  There is no BORN file for '+material +
          '. PHONOPY calculations will process with .NAC. = FALSE\n')
    born_exists = False

if born_exists:
    phonon_file = phonopy.load(
        supercell_matrix=mat_mod.mat_properties_dict['supercell_dim'],
        primitive_matrix='auto',
        unitcell_filename=poscar_path,
        force_sets_filename=force_sets_path,
        is_nac=True,
        born_filename=born_path
    )
else:
    phonon_file = phonopy.load(
        supercell_matrix=mat_mod.mat_properties_dict['supercell_dim'],
        primitive_matrix='auto',
        unitcell_filename=poscar_path,
        force_sets_filename=force_sets_path
    )

phonopy_params = phonopy_funcs.get_phonon_file_data(phonon_file, born_exists)


for key, value in num_mod.numerics_parameters.items():
    if key not in numerics_params:
        numerics_params[key] = value


projection.proj_form_factor(physics_params=physics_params, 
                            numerics_params=numerics_params,
                            phonopy_params=phonopy_params, 
                            file_params=file_params,
                            phonon_file=phonon_file, 
                            c_dict=phys_mod.c_dict,
                            verbose=True)
