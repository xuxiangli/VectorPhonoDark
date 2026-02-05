from numpy import float32
from vectorphonodark import constants as const
from vectorphonodark.projection import BinnedMcalI

input_path = "/Users/jukcoeng/Desktop/Dark_Matter/Vector Space Integration/VectorPhonoDark/"
output_path = "/Users/jukcoeng/Desktop/Dark_Matter/Vector Space Integration/VectorPhonoDark/output/"

mass = 1*10**6
q_max = 2 * mass * (const.VESC + const.VE)
# q_max = 470672.7665
energy_threshold = 1e-3  # eV
q_min = energy_threshold / (const.VESC + const.VE)
f_med = 2
nv_max = 2**7 - 1
nq_max = 2**9 - 1

physics_params = {
    "fdm": (-2*f_med, 0),
    # "q0_fdm": mass * const.V0,  # reference momentum transfer in eV
    "energy_threshold": 1e-3,  # eV
    "energy_bin_width": 1e-3,  # eV
    "mass_dm": mass,
    "mass_sm": const.M_NUCL,
}
numerics_params = {
    "n_bins": 34,
    "l_max": 5,
    "nv_max": nv_max,
    "nq_max": nq_max,
    "v_max": (const.VESC + const.VE) * 1.0,
    "q_max": q_max,
    "log_wavelet_q": True,
    "eps_q": q_min / q_max,
}
file_params = {
    "hdf5": output_path + "mcalI" + ".hdf5",
    "hdf5_group": f"{mass/10**6}MeV/{physics_params['fdm']}_vsdm",
    "hdf5_data": "data",
}
params = {**physics_params, **numerics_params}

binned_mcalI = BinnedMcalI(
    physics_params=physics_params, numerics_params=numerics_params
)
binned_mcalI.project(verbose=True)
binned_mcalI.export_hdf5(
    filename=file_params["hdf5"],
    groupname=file_params["hdf5_group"],
    dataname=file_params["hdf5_data"],
    dtype=float32,
)
