from vectorphonodark import constants as const
from vectorphonodark.projection import FormFactor

input_path = "/Users/jukcoeng/Desktop/Dark_Matter/Vector Space Integration/VectorPhonoDark/"
output_path = "/Users/jukcoeng/Desktop/Dark_Matter/Vector Space Integration/VectorPhonoDark/output/"

mass = 100*10**6  # in eV

physics_params = {
    "energy_threshold": 1e-3,  # eV
    "energy_bin_width": 1e-3,  # eV
    # factor to multiply with Gamma point to get energy cutoff
    # default is 4.0 if not specified
    "energy_max_factor": 1.2,
    "model": "GaAs_hadrophilic",
}
numerics_params = {
    # reference momentum scale in eV
    "q_max": 2 * mass * (const.VESC + const.VE),
    "l_max": 5,
    "n_max": 2**9 - 1,
    "n_grid": (2**9, 25, 25),
    "log_wavelet": True,
    # whether to compute q_cut from Debye-Waller factor
    "q_cut": True,
}
input_params = {
    "material_input": input_path + "inputs/material/GaAs/GaAs.py",
    "physics_model_input": input_path + "inputs/physics_model/hadrophilic.py",
    "numerics_input": input_path + "inputs/numerics/standard.py",
}
file_params = {
    # "csv": output_path
    # + f'form_factor/Al2O3/test/{physics_params["model"]}_{mass/10**6}MeV_{numerics_params["n_grid"]}'
    # + ".csv",
    "hdf5": output_path + "form_factor" + ".hdf5",
    "hdf5_group": f'{physics_params["model"]}/{mass/10**6}MeV/{numerics_params["n_grid"]}',
    "hdf5_data": "data",
}
params = {**physics_params, **numerics_params, **input_params}

form_factor = FormFactor(physics_params=physics_params, numerics_params=numerics_params)
form_factor.project(params=params, verbose=True)
# form_factor.export_csv(filename=file_params['csv'])
form_factor.export_hdf5(
    filename=file_params["hdf5"],
    groupname=file_params["hdf5_group"],
    dataname=file_params["hdf5_data"],
)
