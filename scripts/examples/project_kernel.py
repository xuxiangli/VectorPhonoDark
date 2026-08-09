"""
Project the energy-binned kinematic kernel (BinnedMcalI) onto the wavelet-
harmonic basis.

BinnedMcalI projects the kinematic kernel including the energy-momentum
conservation delta function of the DM-phonon scattering process, binned into
detector energy bins. Its projected coefficients are combined with the VDF and
crystal form factor to yield the scattering rate.

Usage:
    python scripts/examples/project_kernel.py
"""

import logging
from pathlib import Path

from numpy import float32

from vectorphonodark import constants as const
from vectorphonodark import phonopy_funcs
from vectorphonodark.projection import BinnedMcalI


def main():
    # Show the library's progress logs.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- DM model ---
    # DM mass in eV
    mass = 1 * 10**6
    # mediator form factor power
    f_med = 2

    # --- Numerics ---
    # max radial wavelet index for VDF
    nv_max = 2**7 - 1
    # max radial wavelet index for material form factor
    nq_max = 2**9 - 1

    # Use the Debye-Waller cutoff as momentum cutoff (eV)
    material = "GaAs"
    material_input = str(
        project_root / "inputs" / "material" / material / f"{material}.py"
    )
    q_cut = phonopy_funcs.compute_q_cut(material_input)
    q_max = q_cut

    # minimum detectable phonon energy (eV)
    energy_threshold = 1e-3
    # minimum momentum transfer (eV)
    q_min = energy_threshold / (const.VESC + const.VE)

    physics_params = {
        # (a, b) in F_DM^2 = (q/q0)^a (v/v0)^b
        "fdm": (-2 * f_med, 0),
        # reference momentum transfer (eV)
        "q0_fdm": mass * const.V0,
        "energy_threshold": energy_threshold,  # eV
        "energy_bin_width": 1e-3,  # eV
        # DM mass (eV)
        "mass_dm": mass,
        # SM particle mass (eV)
        "mass_sm": const.M_NUCL,
    }
    numerics_params = {
        # number of energy bins
        "n_bins": 38,
        # max multipole
        "l_max": 5,
        "nv_max": nv_max,
        "nq_max": nq_max,
        # velocity upper limit (units of c)
        "v_max": (const.VESC + const.VE) * 1.0,
        "q_max": q_max,
        # use logarithmic radial wavelet in q
        "log_wavelet_q": True,
        # relative IR cutoff for log wavelet
        "eps_q": q_min / q_max,
    }
    file_params = {
        "hdf5": str(output_dir / "mcalI.hdf5"),
        "hdf5_group": f"{mass / 10**6}MeV/{physics_params['fdm']}",
        "hdf5_data": "data",
    }

    binned_mcalI = BinnedMcalI(
        physics_params=physics_params,
        numerics_params=numerics_params,
    )
    binned_mcalI.project()
    binned_mcalI.export_hdf5(
        filename=file_params["hdf5"],
        groupname=file_params["hdf5_group"],
        dataname=file_params["hdf5_data"],
        # save in single precision to reduce file size
        dtype=float32,
    )


if __name__ == "__main__":
    main()
