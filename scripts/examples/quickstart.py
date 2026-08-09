"""The whole calculation, end to end, at small scale.

Projects the Standard Halo Model velocity distribution and the GaAs crystal
form factor onto the wavelet-harmonic basis, contracts them, and prints the
projected reach for one DM mass. Nothing is written to disk -- the two
projections are held in memory and handed straight to Rate, which projects
the kinematic kernel internally.

The resolution here is deliberately coarse (a few wavelets, l_max = 2) so
the script finishes quickly. It is a smoke test of the installation and a
readable map of the workflow, **not** a converged calculation.

Usage:
    python scripts/examples/quickstart.py
"""

import os

# Pin BLAS to one thread before numpy loads: the form-factor projection is
# dominated by tiny per-q-point eigensolves, which multithreaded BLAS makes
# slower. An explicit setting in the environment still wins over this default.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging
from pathlib import Path

import numba
import numpy as np
from scipy import special

from vectorphonodark import Rotation
from vectorphonodark import constants as const
from vectorphonodark.projection import VDF, FormFactor
from vectorphonodark.rate import Rate

# --- Numerics ---
# max multipole
L_MAX = 2
# max radial wavelet index for VDF
NV_MAX = 15
# max radial wavelet index for material form factor
NQ_MAX = 15
# (n_r, n_theta, n_phi) integration grid
VDF_GRID = (16, 30, 30)
FF_GRID = (16, 10, 10)

# DM mass (eV)
MASS_DM = 1.0e6
# mediator form factor: F_DM ~ (q0/q)^f_med
F_MED = 2  # light mediator


@numba.njit
def vdf_shm(v_xyz, v_0, v_e, v_esc, n0) -> float:
    """Standard Halo Model: a truncated Maxwellian, boosted to the lab."""
    v_gal_frame = np.linalg.norm(v_xyz + v_e)
    if v_gal_frame <= v_esc:
        return np.exp(-(v_gal_frame**2) / v_0**2) / n0
    return 0.0


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    project_root = Path(__file__).resolve().parents[2]
    v_max = const.VESC + const.VE

    # --- Step 1: the velocity distribution ---
    # The VDF is projected once, in the frame with +z along the DM wind axis
    # at t = 0, so the Earth's velocity there is exactly (0, 0, v_E). Daily
    # modulation enters later by rotating the crystal.
    v_0, v_esc = const.V0, const.VESC
    n0 = (
        np.pi ** (3 / 2)
        * v_0**2
        * (
            v_0 * special.erf(v_esc / v_0)
            - 2 * v_esc / np.sqrt(np.pi) * np.exp(-(v_esc**2) / v_0**2)
        )
    )
    vdf_physics = {
        "vdf": vdf_shm,
        "vdf_params": {
            "v_0": v_0,
            "v_e": np.array([0.0, 0.0, const.VE]),
            "v_esc": v_esc,
            "n0": n0,
        },
        "model": "SHM",
    }
    vdf_numerics = {
        "v_max": v_max,
        "l_max": L_MAX,
        "n_max": NV_MAX,
        "n_grid": VDF_GRID,
    }
    vdf = VDF(physics_params=vdf_physics, numerics_params=vdf_numerics)
    vdf.project(params={**vdf_physics, **vdf_numerics})

    # --- Step 2: the material form factor ---
    # upper momentum cutoff (eV)
    # chosen here to be the kinematic upper bound for better resolution
    # use the Debye-Waller cutoff instead in project_material_ff.py
    q_max = 2 * MASS_DM * v_max
    # minimum detectable phonon energy (eV)
    energy_threshold = 1e-3
    ff_physics = {
        "energy_threshold": energy_threshold,
        # eV; wider bins than those in the paper
        "energy_bin_width": 1e-2,
        # Energy cutoff, as a multiple of the largest Gamma-point phonon energy
        "energy_max_factor": 1.2,
        # output label
        "model": "GaAs_hadrophilic",
    }
    ff_numerics = {
        "q_max": q_max,
        "l_max": L_MAX,
        "n_max": NQ_MAX,
        "n_grid": FF_GRID,
        # log-spaced radial wavelets
        "log_wavelet": True,
        # minimum momentum transfer for logarithmic wavelet basis (eV)
        "q_min": energy_threshold / v_max,
    }
    ff_inputs = {
        "material_input": str(
            project_root / "inputs" / "material" / "GaAs" / "GaAs.py"
        ),
        "physics_model_input": str(
            project_root / "inputs" / "physics_model" / "hadrophilic.py"
        ),
        "numerics_input": str(project_root / "inputs" / "numerics" / "standard.py"),
    }
    form_factor = FormFactor(physics_params=ff_physics, numerics_params=ff_numerics)
    form_factor.project(params={**ff_physics, **ff_numerics, **ff_inputs})

    # --- Step 3: contract them into a rate ---
    rate = Rate(
        physics_params={
            # (a, b) in F_DM^2 = (q/q0)^a (v/v0)^b
            "fdm": (-2 * F_MED, 0),
            "q0_fdm": MASS_DM * const.V0,
            "mass_dm": MASS_DM,
            "mass_sm": const.M_NUCL,  # hadrophilic: scattering off nuclei
        },
        numerics_params={"l_max": L_MAX, "nv_max": NV_MAX, "nq_max": NQ_MAX},
        vdf=vdf,
        ff=form_factor,
    )

    # One crystal orientation: unrotated, i.e., the crystal axes aligned with
    # the wind frame at t = 0.
    binned = rate.binned_rate(Rotation.identity())
    # summed over energy bins, eV^-2
    gamma = float(sum(binned.values())[0])

    # Reach: the cross section for which 3 events/kg/yr would be seen.
    reach = 3.0 / const.KG_YR / (const.RHO_DM * gamma) * const.INVEV_TO_CM**2

    print(f"\n  DM mass          {MASS_DM / 1e6:g} MeV, light mediator, GaAs")
    print(f"  Rate response    {gamma:.4e} eV^-2  (summed over energy bins)")
    print(f"  Projected reach  {reach:.4e} cm^2   (3 events / kg / yr)")
    print("\n  Coarse resolution -- see the module docstring before quoting this.")


if __name__ == "__main__":
    main()
