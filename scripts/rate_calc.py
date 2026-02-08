import numpy as np
import os
from pathlib import Path

import vsdm

from vectorphonodark import constants as const
from vectorphonodark import utility
from vectorphonodark.projection import VDF, FormFactor
from vectorphonodark.rate import Rate, import_all_form_factors


"""input starts here"""
project_root = Path(__file__).resolve().parent.parent
input_path = str(project_root) + "/"
output_path = str(project_root / "output") + "/"

if not os.path.exists(output_path):
    os.makedirs(output_path)

mass_list = np.array([0.03*10**6, 0.1*10**6, 0.3*10**6, 1*10**6, 3*10**6, 10*10**6, 30*10**6, 100*10**6])
q_max_list = 2*mass_list*(const.VESC + const.VE)
f_med = 2
q0_fdm = mass_list * const.V0  # reference momentum transfer in eV
mass_sm = const.M_NUCL

nv_max = 2**7 - 1
nq_max = 2**9 - 1

l_max = 5
theta_list = list(range(0, 180, 180))
phi_list = list(range(0, 360, 360))
rotationlist = []

events_per_year = 3.0
factor = const.RHO_DM  # additional factor in the rate calculation
verbose = True

file_params_vdf = {
    "hdf5": output_path + "vdf" + ".hdf5",
    "hdf5_group": "SHM/230_240_600",
    "hdf5_data": "data",
    "verbose": True,
}
file_params_form_factor = {
    "hdf5": output_path + "GaAs_hadrophilic" + ".hdf5",
    "hdf5_group": f"log",
    "hdf5_data": "data",
    "verbose": True,
}
"""input ends here"""

vdf = VDF().import_hdf5(
    filename=file_params_vdf["hdf5"],
    groupname=file_params_vdf["hdf5_group"],
    dataname=file_params_vdf["hdf5_data"],
    verbose=file_params_vdf["verbose"],
)

# form_factor = FormFactor().import_hdf5(
#     filename=file_params_form_factor["hdf5"],
#     groupname=file_params_form_factor["hdf5_group"],
#     dataname=file_params_form_factor["hdf5_data"],
#     verbose=file_params_form_factor["verbose"],
# )

form_factors, q_max_list_file = import_all_form_factors(
    filename=file_params_form_factor["hdf5"],
    groupname=file_params_form_factor["hdf5_group"],
    dataname=file_params_form_factor["hdf5_data"],
    verbose=file_params_form_factor["verbose"],
)

for theta in theta_list:
    for phi in phi_list:
        q = 1 / utility.getQ(theta * np.pi / 180, phi * np.pi / 180)
        rotationlist += [q]
wG = vsdm.WignerG(l_max, rotations=rotationlist)

for mass, q_max, q0_fdm in zip(mass_list, q_max_list, q0_fdm):

    physics_params = {
        "fdm": (-2*f_med, 0),
        "q0_fdm": q0_fdm,
        "mass_dm": mass,
        "mass_sm": mass_sm,
    }
    numerics_params = {
        "l_max": l_max,
        "nv_max": nv_max,
        "nq_max": nq_max,
    }

    q_max_file = q_max_list_file[q_max_list_file >= q_max]
    if len(q_max_file) == 0:
        print("\n    Using form factor with largest q_max = "
              f"{q_max_list_file[-1]:.2e} eV for mass {mass/10**6:.2f} MeV.")
        q_max_file = q_max_list_file[-1]
    else:
        print("\n    Using form factor with q_max = "
              f"{q_max_file[0]:.2e} eV for mass {mass/10**6:.2f} MeV "
              f"with optimized q_max = {q_max:.2e} eV.")
        q_max_file = q_max_file[0]

    form_factor = form_factors[q_max_file]

    binned_rate = Rate(
        physics_params=physics_params, 
        numerics_params=numerics_params, 
        vdf=vdf, 
        ff=form_factor,
        verbose=verbose,
    )

    rate_r = sum(binned_mu_R for binned_mu_R in binned_rate.binned_mu_R(wG=wG).values())

    for i_rot in range(len(rotationlist)):
        reach = (
            events_per_year / const.KG_YR / (factor * float(rate_r[i_rot])) 
            * const.inveV_to_cm**2
        )
        print(f"Mass {mass/10**6} MeV, f_med {f_med}, rotation {i_rot}:")
        print(
            f"    Projected reach for {events_per_year} events "
            f"per year: {float(reach):.4e} cm^2"
        )
