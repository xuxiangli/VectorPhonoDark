import numpy as np

import vsdm

from vectorphonodark import constants as const
from vectorphonodark import utility


"""Inputs start here"""
output_path = '/Users/jukcoeng/Desktop/Dark_Matter/Vector Space Integration/VectorPhonoDark/'

mass_list = np.array([10**5, 10**6, 10**7, 10**8])  # in eV
fdm_list = [0, 2]
q0 = mass_list*const.V0  # reference momentum transfer in eV
events_per_year = 3.0

l_max = 5
nv_max = 31
nq_max = 127

physics_params = {
    'threshold': 1e-3           # eV
}
numerics_params = {
    'energy_bin_width': 1e-3,   # eV
    'energy_bin_num': 34
}
form_factor_path = {
    10**5: output_path+'output/form_factor/GaAs/100keV/GaAs_hadrophilic_100keV_512_25_25',
    10**6: output_path+'output/form_factor/GaAs/1MeV/GaAs_hadrophilic_1MeV_512_25_25',
    10**7: output_path+'output/form_factor/GaAs/10MeV/GaAs_hadrophilic_10MeV_512_25_25',
    10**8: output_path+'output/form_factor/GaAs/100MeV/GaAs_hadrophilic_100MeV_128_25_25'
}
file_params = {
    'vdf_path': output_path+'output/vdf/shm_230_240_600_128_180_180',
    'form_factor_path': form_factor_path,
    'mcalI_path': output_path+'output/mcalI/mcalI_5_31_127'
}
"""Inputs end here"""


v_max = (const.VESC + const.VE) * 1.0
basis_v = dict(u0=v_max, type='wavelet', uMax=v_max)

vdf = vsdm.Fnlm(basis_v, f_type='gX', use_gvar=False)
vdf.importFnlm_csv(file_params['vdf_path']+'.csv')

energy_threshold = physics_params['threshold']
energy_bin_width = numerics_params['energy_bin_width']
energy_bin_num = numerics_params['energy_bin_num']

rotationlist = []
for theta in range(0, 180, 180):
    for phi in range(0, 360, 360):
        q = 1/utility.getQ(theta * np.pi/180, phi * np.pi/180)
        rotationlist += [q]
wG = vsdm.WignerG(l_max, rotations=rotationlist)

for i_mass, mass in enumerate(mass_list):

    q_max = 2*mass*(const.VESC + const.VE)
    # q_max = 470672.7665
    basis_q = dict(u0=q_max, type='wavelet', uMax=q_max)

    form_factor_list = np.empty(energy_bin_num, dtype=object)
    for i_bin in range(energy_bin_num):
        form_factor = vsdm.Fnlm(basis_q, f_type='fs2', use_gvar=False)
        form_factor.importFnlm_csv(
            file_params['form_factor_path'][mass]+'_bin_'+str(i_bin)+'.csv')
        form_factor_list[i_bin] = form_factor

    for fdm in fdm_list:

        rate_r = np.zeros(len(rotationlist))
        factor = const.RHO_DM * (q0[i_mass]/const.Q_BOHR)**(2*fdm)

        for i_bin in range(energy_bin_num):
            energy = energy_threshold + (i_bin + 0.5)*energy_bin_width
            dm_model = dict(mX=mass, fdm=fdm, mSM=const.M_NUCL, DeltaE=energy)
            mcalI = vsdm.McalI(basis_v, basis_q, dm_model,
                               mI_shape=(l_max+1, nv_max+1, nq_max+1),
                               use_gvar=False, do_mcalI=False)
            mcalI.importMcalI(hdf5file=file_params['mcalI_path']+'.hdf5',
                              modelName=f"({i_bin}, {mass}, {fdm})",
                            #   modelName=f"{mass:.3e}/{fdm}/{i_bin}",
                              d_pair=['Ilvq_mean'])

            rate = vsdm.RateCalc(vdf, form_factor_list[i_bin], mcalI,
                                 use_gvar=False, sparse=False)

            rate_r += rate.mu_R(wG, use_vecK=False)

            del mcalI, rate

        rate_r *= factor

        for i_rot in range(len(rotationlist)):
            reach = events_per_year / const.KG_YR / \
                rate_r[i_rot] * const.inveV_to_cm**2
            print(f"Mass {mass/10**6} MeV, fn {fdm}, rotation {i_rot}:")
            print(
                f"    Projected reach for {events_per_year} events per year: {reach} cm^2")
