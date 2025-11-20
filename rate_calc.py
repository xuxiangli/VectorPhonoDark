import numpy as np

import vsdm

from src import constants as const
from src import utility


"""Inputs start here"""
output_path = './'

mass_list = [10**6] # in eV
fn_list = [0]

l_max = 5
nv_max = 31
nq_max = 31

physics_params = {
    'threshold': 1e-3           # eV
    }
numerics_params = {
    'energy_bin_width': 1e-3,   # eV
    'energy_bin_num': 34,
    'q_cut': True               # whether to compute q_cut from Debye-Waller factor
    }
form_factor_path = {
    10**5: output_path+'out/form_factor/GaAs_100keV_hadrophilic_64_25_25',
    10**6: output_path+'out/form_factor/GaAs_1MeV_hadrophilic_64_25_25'
    }
file_params = {
    'vdf_path': output_path+'out/vdf/shm_230_240_600_128_25_25',
    'form_factor_path': form_factor_path,
    'mcalI_path': output_path+'out/mcalI/mcalI_5_31_31'
    }
"""Inputs end here"""


v_max = (const.VESC + const.VE) * 1.0
basisV = dict(u0=v_max, type='wavelet', uMax=v_max)

vdf = vsdm.Fnlm(basisV, f_type='gX', use_gvar=False)
vdf.importFnlm_csv(file_params['vdf_path']+'.csv')

energy_threshold    = physics_params['threshold']
energy_bin_width    = numerics_params['energy_bin_width']
energy_bin_num      = numerics_params['energy_bin_num']

rotationlist = []
for theta in range(0, 180, 180):
    for phi in range(0, 360, 360):
        q = 1/utility.getQ(theta * np.pi/180, phi * np.pi/180) 
        rotationlist += [q] 
wG = vsdm.WignerG(l_max, rotations=rotationlist)


for mass in mass_list:

    q_max = 2*mass*(const.VESC + const.VE)
    basisQ = dict(u0=q_max, type='wavelet', uMax=q_max)

    form_factor_list = np.empty(energy_bin_num, dtype=object)
    for i in range(energy_bin_num):
        form_factor = vsdm.Fnlm(basisQ, f_type='fs2', use_gvar=False)
        form_factor.importFnlm_csv(file_params['form_factor_path'][mass]+'_bin_'+str(i)+'.csv')
        form_factor_list[i] = form_factor

    for fn in fn_list:

        rate_R = np.zeros(len(rotationlist))

        for i in range(energy_bin_num):
            deltaE = energy_threshold + (i + 0.5)*energy_bin_width
            dmModel = dict(mX=mass, fdm_n=fn, mSM=const.M_NUCL, DeltaE=deltaE)
            mcalI = vsdm.McalI(basisV, basisQ, dmModel, 
                                    mI_shape=(l_max+1, nv_max+1, nq_max+1), 
                                    use_gvar=False, do_mcalI=False)
            mcalI.importMcalI(hdf5file=file_params['mcalI_path']+'.hdf5', modelName=f"({i}, {mass}, {fn})", d_pair=['Ilvq_mean'])

            rates = vsdm.RateCalc(vdf, form_factor_list[i], mcalI,
                                        use_gvar=False, sparse=False)

            rate_R += rates.mu_R(wG)
        
        for i_rot in range(len(rotationlist)):
            reach = 3.0/(const.KG_YR*const.RHO_DM)*v_max*q_max**4/rate_R[i_rot]*const.inveV_to_cm**2
            print(f"Mass {mass} eV, fn {fn}, rotation {i_rot}: Projected reach for 3 events per year: {reach} cm^2")