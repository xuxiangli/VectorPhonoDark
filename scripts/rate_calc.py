import numpy as np

import vsdm

from vectorphonodark import constants as const
from vectorphonodark import utility
from vectorphonodark.projection import VDF, FormFactor, BinnedMcalI
from vectorphonodark.rate import BinnedRate


output_path = "/Users/jukcoeng/Desktop/Dark_Matter/Vector Space Integration/VectorPhonoDark/output/"

mass = 10**6
fdm = 2
q0 = mass * const.V0  # reference momentum transfer in eV
q_scaling = fdm
q_factor = (q0 / const.Q_BOHR) ** (2 * q_scaling)
events_per_year = 3.0
factor = const.RHO_DM * q_factor

nv_list = []  # list(range(2**7))
nq_list = (
    list(range(2**7))
    + list(range(2**7, 2**7 + 2))
    + list(range(2**8, 2**8 + 4))
    + list(range(2**9, 2**9 + 8))
)

l_max = 5
rotationlist = []
for theta in range(0, 180, 180):
    for phi in range(0, 360, 360):
        q = 1 / utility.getQ(theta * np.pi / 180, phi * np.pi / 180)
        rotationlist += [q]
wG = vsdm.WignerG(l_max, rotations=rotationlist)

file_params_vdf = {
    # 'csv': output_path+'vdf/shm_230_240_600_128_180_180_1'+'.csv',
    "hdf5": output_path + "test" + ".hdf5",
    "hdf5_group": "vdf/SHM/230_240_600/(128, 180, 180)",
    "hdf5_data": "data",
}
file_params_form_factor = {
    # 'csv': output_path+'form_factor/GaAs/test/GaAs_hadrophilic_1MeV_128_25_25'+'.csv',
    "hdf5": output_path + "test" + ".hdf5",
    "hdf5_group": f"form_factor/GaAs_hadrophilic/{mass/10**6}MeV/(128, 25, 25)/",
    "hdf5_data": "data",
}
file_params_mcalI = {
    "hdf5": output_path + "test" + ".hdf5",
    "hdf5_group": f"mcalI/{mass/10**6}MeV/({fdm},0)/",
    "hdf5_data": "data",
}

vdf = VDF(physics_params={}, numerics_params={})
# vdf.import_csv(filename=file_params_vdf['csv'])
vdf.import_hdf5(
    filename=file_params_vdf["hdf5"],
    groupname=file_params_vdf["hdf5_group"],
    dataname=file_params_vdf["hdf5_data"],
)

form_factor = FormFactor(physics_params={}, numerics_params={})
# form_factor.import_csv(filename=file_params_form_factor['csv'])
form_factor.import_hdf5(
    filename=file_params_form_factor["hdf5"],
    groupname=file_params_form_factor["hdf5_group"],
    dataname=file_params_form_factor["hdf5_data"],
)

mcalI = BinnedMcalI(physics_params={}, numerics_params={})
mcalI.import_hdf5(
    filename=file_params_mcalI["hdf5"],
    groupname=file_params_mcalI["hdf5_group"],
    dataname=file_params_mcalI["hdf5_data"],
)

binned_rate = BinnedRate(
    vdf,
    form_factor,
    mcalI,
    nv_list=nv_list,
    nq_list=nq_list,
)
rate_r = sum(binned_mu_R for binned_mu_R in binned_rate.binned_mu_R(wG=wG).values())

for i_rot in range(len(rotationlist)):
    reach = (
        events_per_year / const.KG_YR / (factor * rate_r[i_rot]) * const.inveV_to_cm**2
    )
    print(f"Mass {mass/10**6} MeV, fn {q_scaling}, rotation {i_rot}:")
    print(
        f"    Projected reach for {events_per_year} events per year: {reach:.4e} cm^2"
    )
