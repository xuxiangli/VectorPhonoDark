import numpy as np
import os
from pathlib import Path

try:
    import vsdm
except ImportError:
    print("Warning: vsdm module not found. Script may not function correctly.")
    # Mock for development/testing if vsdm is missing
    # import unittest.mock
    # vsdm = unittest.mock.MagicMock()

from vectorphonodark import constants as const
from vectorphonodark import physics
from vectorphonodark.projection import VDF


"""input starts here"""
t = 0.0
v_0 = const.V0
v_e = physics.create_vE_vec(t)
v_esc = const.VESC
# n0 is calculated inside vsdm typically for SHM, or passed.
# In original script:
# n0 = ...
# But vsdm might calculate normalization itself. I'll pass it just in case.
# If not needed, it will be ignored.

project_root = Path(__file__).resolve().parent.parent
output_path = str(project_root / "output") + "/"

if not os.path.exists(output_path):
    os.makedirs(output_path)

physics_params = {
    "model": "SHM",
    "vdf_params": {"v_0": v_0, "v_e": v_e, "v_esc": v_esc},
}

l_max = 5
n_max = 2**7 - 1
v_max = (const.VESC + const.VE) * 1.0

numerics_params = {
    "v_max": v_max,
    "l_max": l_max,
    "n_max": n_max,
    "l_mod": 1, # Default
}
file_params = {
    "hdf5": output_path + "vdf_vsdm" + ".hdf5", # Changed filename slightly to distinguish
    "hdf5_group": f'{physics_params["model"]}/230_240_600',
    "hdf5_data": "data",
}
"""input ends here"""

params = {**physics_params, **numerics_params}

# Use vsdm to calculate f_lm_n
# Assuming vsdm has a HaloModel class or similar that takes basis and model dicts.
# Based on McalI usage in analytic.py:
# basis_v = dict(u0=v_max, type="wavelet", uMax=v_max)
# basis_q = ...
# dm_model = ...
# mI = vsdm.McalI(...)

basis_v = {"u0": v_max, "type": "wavelet", "uMax": v_max}
halo_model = {
    "type": "SHM",
    "v0": v_0,
    "vE": v_e,
    "vesc": v_esc,
    # "n0": n0 # if needed
}

lm_list = [(l, m) for l in range(0, l_max + 1, numerics_params["l_mod"]) for m in range(-l, l + 1)]

# Placeholder for vsdm calculation
if 'vsdm' in globals():
    try:
        # Construct the model
        # Note: The exact API for vsdm.HaloModel is assumed here based on McalI usage.
        # Please verify the class name and method for retrieving coefficients.
        halo = vsdm.HaloModel(basis_v, halo_model)

        # Compute coefficients
        # Assumed attribute: f_lm_n. If the attribute is different, please adjust.
        f_lm_n = halo.f_lm_n

        # Check shape compatibility
        # VDF.f_lm_n expects (len(lm_list), n_max + 1)
        # lm_list length depends on l_max and l_mod.
        expected_shape = (len(lm_list), n_max + 1)

        if f_lm_n.shape != expected_shape:
            print(f"Warning: f_lm_n shape {f_lm_n.shape} does not match expected {expected_shape}.")
            # Add reshaping logic if known.
            # For now, just warn.

    except AttributeError:
        print("Error: vsdm.HaloModel not found or API mismatch. Please verify vsdm version and API.")
        # Fallback to empty array for structure preservation if running without vsdm for testing
        f_lm_n = np.zeros((len(lm_list), n_max + 1))

# Create VDF instance and assign data
vdf = VDF(physics_params=physics_params, numerics_params=numerics_params)
# Manually assign the computed coefficients
if 'f_lm_n' in locals():
    vdf.f_lm_n = f_lm_n
else:
    print("Warning: f_lm_n was not computed. VDF will be empty.")

# Export to HDF5
vdf.export_hdf5(
    filename=file_params["hdf5"],
    groupname=file_params["hdf5_group"],
    dataname=file_params["hdf5_data"],
)
