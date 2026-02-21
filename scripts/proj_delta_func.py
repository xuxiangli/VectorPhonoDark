"""
Project the energy-binned kinematic kernel (BinnedMcalI) onto the spherical
wavelet basis.

BinnedMcalI encodes the energy-momentum conservation delta function of the
DM-phonon scattering process, binned into detector energy bins.  Its wavelet
coefficients are combined with the VDF and form factor to yield the
differential scattering rate.

Usage:
    python proj_delta_func.py
"""

from numpy import float32
from pathlib import Path

from vectorphonodark import constants as const
from vectorphonodark.projection import BinnedMcalI


def main():
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- DM model ---
    mass = 1 * 10**6                  # DM mass in eV
    q_max = 2 * mass * (const.VESC + const.VE)  # kinematic momentum upper limit (eV)
    # q_max = 470672.7665
    energy_threshold = 1e-3           # eV; minimum detectable phonon energy
    q_min = energy_threshold / (const.VESC + const.VE)  # corresponding minimum momentum (eV)
    f_med = 2                         # mediator form factor power

    # --- Numerics ---
    nv_max = 2**7 - 1                 # max radial wavelet index for VDF
    nq_max = 2**9 - 1                 # max radial wavelet index for q

    physics_params = {
        "fdm": (-2 * f_med, 0),       # (numerator power, denominator power) of F_med
        "q0_fdm": mass * const.V0,    # reference momentum transfer in eV
        "energy_threshold": 1e-3,     # eV
        "energy_bin_width": 1e-3,     # eV
        "mass_dm": mass,              # DM mass (eV)
        "mass_sm": const.M_NUCL,      # SM target mass (eV)
    }
    numerics_params = {
        "n_bins": 34,                 # number of energy bins
        "l_max": 5,                   # max angular momentum quantum number
        "nv_max": nv_max,
        "nq_max": nq_max,
        "v_max": (const.VESC + const.VE) * 1.0,  # velocity upper limit (eV)
        "q_max": q_max,               # momentum upper limit (eV)
        "log_wavelet_q": True,        # use logarithmic radial wavelet in q
        "eps_q": q_min / q_max,       # relative IR cutoff for log wavelet
    }
    file_params = {
        "hdf5": str(output_dir / "mcalI.hdf5"),
        "hdf5_group": f"{mass / 10**6}MeV/{physics_params['fdm']}_vsdm",
        "hdf5_data": "data",
    }

    params = {**physics_params, **numerics_params}

    binned_mcalI = BinnedMcalI(
        physics_params=physics_params,
        numerics_params=numerics_params,
    )
    binned_mcalI.project(verbose=True)
    binned_mcalI.export_hdf5(
        filename=file_params["hdf5"],
        groupname=file_params["hdf5_group"],
        dataname=file_params["hdf5_data"],
        dtype=float32,                # save in single precision to reduce file size
    )


if __name__ == "__main__":
    main()
