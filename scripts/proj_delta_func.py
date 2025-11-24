import os
import time

import phonopy
import vsdm

from vectorphonodark import constants as const
from vectorphonodark import phonopy_funcs
from vectorphonodark import utility
from vectorphonodark import physics


"""Inputs start here"""
input_path = ''
output_path = ''

mass = 10**5

l_max = 5
nv_max = 31
nq_max = 31

physics_params = {
    'threshold': 1e-3           # eV
    }
numerics_params = {
    'energy_bin_width': 1e-3,   # eV
    'energy_max': 0.10723,      # eV, maximal energy for all materials considered
    'q_cut': True               # whether to compute q_cut from Debye-Waller factor
    }
file_params = {
    'modelname': 'GaAs',
    'hdf5name': output_path+'output/mcalI/mcalI_5_31_31'
    }

material_input = input_path+'inputs/material/GaAs/GaAs.py'
"""Inputs end here"""

mat_input_mod_name = os.path.splitext(os.path.basename(material_input))[0]
mat_mod = utility.import_file(mat_input_mod_name, os.path.join(material_input))
material = mat_mod.material

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
    print('  There is no BORN file for '+material+'. PHONOPY calculations will process with .NAC. = FALSE\n')
    born_exists = False
    
if born_exists: 
    phonon_file = phonopy.load(
                    supercell_matrix    = mat_mod.mat_properties_dict['supercell_dim'],
                    primitive_matrix    = 'auto',
                    unitcell_filename   = poscar_path,
                    force_sets_filename = force_sets_path,
                    is_nac              = True,
                    born_filename       = born_path
                    )
else:
    phonon_file = phonopy.load(
                        supercell_matrix    = mat_mod.mat_properties_dict['supercell_dim'],
                        primitive_matrix    = 'auto',
                        unitcell_filename   = poscar_path,
                        force_sets_filename = force_sets_path
                        )

print("\n    Starting projection of delta function...")
start_total_time = time.time()

v_max = (const.VESC + const.VE) * 1.0
basis_v = dict(u0=v_max, type='wavelet', uMax=v_max)

q_max = 2*mass*(const.VESC + const.VE)
if numerics_params['q_cut']:
    phonopy_params = phonopy_funcs.get_phonon_file_data(phonon_file, born_exists)
    q_cut = physics.compute_q_cut(phonon_file, phonopy_params['atom_masses'])
    if q_cut < q_max:
        q_max = q_cut
        file_params['hdf5name'] += '_' + file_params['modelname'] + '_qcut'
        print(f"    Adjusted q_max to {q_max:.4f} eV due to Debye Waller factor.")
basis_q = dict(u0=q_max, type='wavelet', uMax=q_max)

energy_threshold    = physics_params['threshold']
energy_bin_width    = numerics_params['energy_bin_width']
energy_max          = numerics_params['energy_max']
energy_bin_num      = int((energy_max - energy_threshold)/energy_bin_width) + 1

for fn in [0, 2]:
    print(f"\n    Projecting f_n = {fn}...")
    for i_bin in range(energy_bin_num):
        if i_bin % (energy_bin_num // 5 + 1) == 0:
                print(f"      Projecting energy bin {i_bin}/{energy_bin_num-1}...")
        energy = energy_threshold + (i_bin + 0.5)*energy_bin_width
        dm_model = dict(mX=mass, fdm_n=fn, mSM=const.M_NUCL, DeltaE=energy)

        mI = vsdm.McalI(basis_v, basis_q, dm_model, 
                        mI_shape=(l_max+1, nv_max+1,nq_max+1), 
                        use_gvar=False, do_mcalI=True)
        mI.writeMcalI(hdf5file=file_params['hdf5name']+'.hdf5', modelName=(i_bin, mass, fn))

print("    Projection completed.")
print(f"\n    Coefficients saved to {file_params['hdf5name']}.hdf5")
end_total_time = time.time()
print(f"    Total projection time: {end_total_time - start_total_time:.2f} seconds.")