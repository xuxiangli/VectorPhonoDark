from vectorphonodark import constants as const
from vectorphonodark.projection import BinnedMcalI

input_path = (
    "/Users/jukcoeng/Desktop/Dark_Matter/Vector Space Integration/VectorPhonoDark/"
)
output_path = "/Users/jukcoeng/Desktop/Dark_Matter/Vector Space Integration/VectorPhonoDark/output/"

mass = 10**6
q_max = 2 * mass * (const.VESC + const.VE)
nv_list = list(range(2**7))
nq_list = list(range(2**10)) + [2**p for p in range(10, 13)]

physics_params = {
    "fdm": 2,
    "energy_threshold": 1e-3,  # eV
    "energy_bin_width": 1e-3,  # eV
    "mass_dm": mass,
    "mass_sm": const.M_NUCL,
}
numerics_params = {
    "n_bins": 34,
    "l_max": 5,
    "nv_list": nv_list,
    "nq_list": nq_list,
    "v_max": (const.VESC + const.VE) * 1.0,
    "q_max": q_max,
}
file_params = {
    "hdf5": output_path + "test",
    "hdf5_group": f"mcalI/{mass/10**6}MeV/({physics_params['fdm']},0)/new/",
    "hdf5_data": "data",
}
params = {**physics_params, **numerics_params}

binned_mcalI = BinnedMcalI(
    physics_params=physics_params, numerics_params=numerics_params
)
binned_mcalI.project(params=params, verbose=True)
binned_mcalI.export_hdf5(
    filename=file_params["hdf5"],
    groupname=file_params["hdf5_group"],
    dataname=file_params["hdf5_data"],
)
