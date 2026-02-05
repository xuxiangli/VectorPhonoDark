import numpy as np

import vsdm

from vectorphonodark import constants as const
from vectorphonodark import utility
from vectorphonodark.projection import VDF, FormFactor, BinnedMcalI
from vectorphonodark.rate import BinnedRate


output_path = "/Users/jukcoeng/Desktop/Dark_Matter/Vector Space Integration/VectorPhonoDark/output/"

mass = 10*10**6
f_med = 2
q0 = mass * const.V0  # reference momentum transfer in eV
events_per_year = 3.0
factor = const.RHO_DM

nv_list = list(range(2**7))
nq_list = list(range(2**7))

l_max = 5
theta_list = list(range(0, 180, 180))
phi_list = list(range(0, 360, 360))
rotationlist = []

physics_params = {
    "fdm": (-2*f_med, 0),
    "q0_fdm": q0,
    "mass_dm": mass,
    "mass_sm": const.M_NUCL,
}
numerics_params = {
    "l_max": l_max,
    "nv_list": nv_list,
    "nq_list": nq_list,
}

file_params_vdf = {
    # 'csv': output_path+'vdf/shm_230_240_600_128_180_180_1'+'.csv',
    "hdf5": output_path + "vdf" + ".hdf5",
    "hdf5_group": "vdf/SHM/230_240_600/(128, 180, 180)",
    "hdf5_data": "data",
    "verbose": True,
}
file_params_form_factor = {
    # 'csv': output_path+'form_factor/GaAs/test/GaAs_hadrophilic_1MeV_128_25_25'+'.csv',
    "hdf5": output_path + "mismatch/test" + ".hdf5",
    # "hdf5_group": f"form_factor/GaAs_hadrophilic/{mass/10**6}MeV/(512, 25, 25)",
    "hdf5_group": f"form_factor/GaAs_hadrophilic/100.0MeV/(512, 25, 25)",
    "hdf5_data": "data",
    "verbose": True,
}

vdf = VDF(physics_params={}, numerics_params={})
# vdf.import_csv(filename=file_params_vdf['csv'])
vdf.import_hdf5(
    filename=file_params_vdf["hdf5"],
    groupname=file_params_vdf["hdf5_group"],
    dataname=file_params_vdf["hdf5_data"],
    verbose=file_params_vdf["verbose"],
)

form_factor = FormFactor(physics_params={}, numerics_params={})
# form_factor.import_csv(filename=file_params_form_factor['csv'])
form_factor.import_hdf5(
    filename=file_params_form_factor["hdf5"],
    groupname=file_params_form_factor["hdf5_group"],
    dataname=file_params_form_factor["hdf5_data"],
    verbose=file_params_form_factor["verbose"],
)

binned_rate = BinnedRate(
    physics_params=physics_params, 
    numerics_params=numerics_params, 
    vdf=vdf, 
    ff=form_factor
)

for theta in theta_list:
    for phi in phi_list:
        q = 1 / utility.getQ(theta * np.pi / 180, phi * np.pi / 180)
        rotationlist += [q]
wG = vsdm.WignerG(l_max, rotations=rotationlist)

rate_r = sum(binned_mu_R for binned_mu_R in binned_rate.binned_mu_R(wG=wG).values())

for i_rot in range(len(rotationlist)):
    reach = (
        events_per_year / const.KG_YR / (factor * float(rate_r[i_rot])) * const.inveV_to_cm**2
    )
    print(f"Mass {mass/10**6} MeV, f_med {f_med}, rotation {i_rot}:")
    print(
        f"    Projected reach for {events_per_year} events "
        f"per year: {float(reach):.4e} cm^2"
    )
