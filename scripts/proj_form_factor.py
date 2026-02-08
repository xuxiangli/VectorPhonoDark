import os
from pathlib import Path
from vectorphonodark import constants as const
from vectorphonodark.projection import FormFactor


"""input starts here"""
project_root = Path(__file__).resolve().parent.parent
input_path = str(project_root) + "/"
output_path = str(project_root / "output") + "/"

if not os.path.exists(output_path):
    os.makedirs(output_path)

# q_max_list = [2.5*10**2, 1.0*10**3, 2.5*10**3, 1.0*10**4, 2.5*10**4, 1.0*10**5, 2.5*10**5]
q_max_list = [470672.7665347568]

for q_max in q_max_list:

    physics_params = {
        "energy_threshold": 1e-3,  # eV
        "energy_bin_width": 1e-3,  # eV
        # factor to multiply with Gamma point to get energy cutoff
        # default: 4.0
        "energy_max_factor": 1.2,
        "model": "Al2O3_dark_photon",
    }
    numerics_params = {
        "q_max": q_max,
        "l_max": 5,
        "n_max": 2**9 - 1,
        "n_grid": (2**9, 25, 25),
        "log_wavelet": True,
    }
    input_params = {
        "material_input": input_path + "inputs/material/Al2O3/Al2O3.py",
        "physics_model_input": input_path + "inputs/physics_model/dark_photon.py",
        "numerics_input": input_path + "inputs/numerics/standard.py",
    }
    file_params = {
        "hdf5": output_path + f"{physics_params['model']}" + ".hdf5",
        "hdf5_group": f'log/{q_max} eV',
        "hdf5_data": "data",
    }
    """input ends here"""

    params = {**physics_params, **numerics_params, **input_params}

    form_factor = FormFactor(
        physics_params=physics_params, 
        numerics_params=numerics_params
    )
    form_factor.project(params=params, verbose=True)
    form_factor.export_hdf5(
        filename=file_params["hdf5"],
        groupname=file_params["hdf5_group"],
        dataname=file_params["hdf5_data"],
    )
