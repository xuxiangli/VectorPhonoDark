"""Physics projection layer: VDF, FormFactor, McalI, BinnedMcalI.

Each projects its integrand onto the wavelet-harmonic basis. The
storage/IO base classes live in :mod:`vectorphonodark.fnlm`; the mesh and
integrator helpers in :mod:`vectorphonodark.mesh`.
"""

import logging
import time

import h5py
import numpy as np
import numpy.typing as npt
from numba.core.errors import NumbaError

from . import constants as const
from . import phonopy_funcs
from .fnlm import HDF5_FORMAT_VERSION, BinnedFnlm, Fnlm
from .mesh import (
    _validate_radial_grid,
    gen_mesh_ylm_jacob,
    gen_mesh_ylm_jacob_log,
    proj_get_f_lm_n,
)

logger = logging.getLogger(__name__)

# from . import analytic
try:
    from . import analytic_cy as analytic
except ImportError:
    logger.warning("Could not load Cython module, falling back to Python version.")
    from . import analytic

# The kinematic-kernel matrix I_l(nv, nq) of whichever backend was selected
# above; re-exported as vectorphonodark.kin_matrix.
kin_matrix = analytic.kin_matrix


class VDF(Fnlm):
    """
    Velocity Distribution Function class.

    Parameters
    ----------
    physics_params : dict or None
        Physics parameters required for VDF projection.
            - vdf: the velocity distribution function.
            - vdf_params: parameters for the velocity distribution function.
        Optional keys:
            - model: name of the VDF model.

    numerics_params : dict or None
        Numerical parameters required for VDF projection.
            - v_max: reference velocity scale.
            - l_max: maximum angular momentum quantum number.
            - n_max: maximum radial quantum number.
        Optional keys:
            - l_mod: denotes parity of l values (1 for all, 2 for even only).
    """

    def __init__(
        self,
        physics_params: dict | None = None,
        numerics_params: dict | None = None,
    ):
        if physics_params is None:
            physics_params = {}
        if numerics_params is None:
            numerics_params = {}

        l_max = numerics_params.get("l_max", -1)
        l_mod = numerics_params.get("l_mod", 1)
        n_max = numerics_params.get("n_max", -1)
        super().__init__(l_max=l_max, l_mod=l_mod, n_max=n_max)

        self.vdf = physics_params.get("vdf", None)
        self.vdf_params = physics_params.get("vdf_params", {})

        self.v_max = numerics_params.get("v_max", 1.0)

        self.f_lm_n = np.array([])
        for key, value in self.vdf_params.items():
            self.info["vdf_param_" + key] = value
        self.info["v_max"] = self.v_max

        if "model" in physics_params:
            self.info["model"] = physics_params["model"]

    def import_hdf5(self, filename: str, groupname: str, dataname: str = "data"):
        super().import_hdf5(filename, groupname, dataname)
        keys_to_load = [
            # "l_max",
            # "l_mod",
            # "n_max",
            "v_max"
        ]
        for key in keys_to_load:
            if key in self.info:
                setattr(self, key, self.info[key])
            else:
                raise KeyError(f"Key '{key}' not found in HDF5 info.")

        return self

    def project(self, params: dict):
        """
        Project the VDF onto basis functions by quadrature on a spherical grid.

        Parameters
        ----------
        params : dict
            Projection parameters. Required keys:
                - n_grid: tuple of ints (n_a, n_b, n_c) for the projection grid.
                  n_a (radial) must be a power of two greater than n_max.
        """

        if self.vdf is None:
            raise ValueError(
                "VDF function not set. Please provide a valid velocity "
                "distribution function."
            )

        model = self.info.get("model")
        logger.info(f"\n=== VDF projection{f' -- {model}' if model else ''} ===")
        start_total_time = time.time()

        l_max = self.l_max
        l_mod = self.l_mod
        n_max = self.n_max

        vdf = self.vdf
        vdf_params = self.vdf_params
        v_max = self.v_max

        n_a, n_b, n_c = params["n_grid"]
        self.info["n_grid"] = params["n_grid"]
        _validate_radial_grid(n_a, n_max)

        logger.info(f"    Parameters for VDF: {vdf_params}")
        logger.info(f"    Reference velocity v_max = {v_max:.2e}.")
        logger.info(
            f"    Projecting onto basis with l_max={l_max}, "
            f"l_mod={l_mod}, n_max={n_max}."
        )
        logger.info(f"    Grid size: n_a={n_a}, n_b={n_b}, n_c={n_c}")
        logger.info("Generating grids and evaluating the VDF...")
        start_time = time.time()

        # Prepare grid points and function values
        lm_list = np.array(
            [
                (ell, m)
                for ell in range(0, l_max + 1, l_mod)
                for m in range(-ell, ell + 1)
            ],
            dtype=np.int64,
        )
        v_xyz_list, y_lm_vals, jacob_vals = gen_mesh_ylm_jacob(
            lm_list, v_max, n_a, n_b, n_c
        )

        try:
            vdf_vals_flat = vdf(v_xyz_list, **vdf_params)
            if not isinstance(vdf_vals_flat, np.ndarray) or vdf_vals_flat.shape != (
                len(v_xyz_list),
            ):
                raise ValueError(
                    "VDF output is not a 1D numpy array of the correct length"
                )
            vdf_vals = vdf_vals_flat.reshape(n_a, n_b, n_c)
        except (TypeError, ValueError, NumbaError):
            vdf_vals = np.array(
                [vdf(v_vec, **vdf_params) for v_vec in v_xyz_list]
            ).reshape(n_a, n_b, n_c)
        del v_xyz_list

        end_time = time.time()
        logger.debug(
            f"VDF calculation completed in {end_time - start_time:.2f} seconds."
        )
        logger.info("Projecting VDF onto basis functions...")

        # Project vdf onto basis functions
        self.f_lm_n = proj_get_f_lm_n(
            n_max=n_max,
            lm_list=lm_list,
            func_vals=vdf_vals,
            y_lm_vals=y_lm_vals,
            jacob_vals=jacob_vals,
            n_a=n_a,
        )
        del vdf_vals, y_lm_vals, jacob_vals

        end_total_time = time.time()
        logger.info(f"=== Done in {end_total_time - start_total_time:.1f} s ===")


class FormFactor(BinnedFnlm):
    """
    Material Form Factor class.

    Parameters
    ----------
    physics_params : dict or None
        Physics parameters required for material form factor.
            - energy_threshold: minimum energy transfer.
            - energy_bin_width: width of each energy bin.
        Optional keys:
            - energy_max_factor: factor to determine maximum energy.
            - model: name of the material form factor model.

    numerics_params : dict or None
        Numerical parameters required for material form factor projection.
            - q_max: reference momentum scale.
            - l_max: maximum angular momentum quantum number.
            - n_max: maximum radial quantum number.
        Optional keys:
            - l_mod: denotes parity of l values (1 for all, 2 for even only).
            - log_wavelet: whether to use log wavelet mesh. Default is True.
    """

    def __init__(
        self,
        physics_params: dict | None = None,
        numerics_params: dict | None = None,
    ):
        if physics_params is None:
            physics_params = {}
        if numerics_params is None:
            numerics_params = {}

        l_max = numerics_params.get("l_max", -1)
        l_mod = numerics_params.get("l_mod", 1)
        n_max = numerics_params.get("n_max", -1)
        super().__init__(n_bins=0, l_max=l_max, l_mod=l_mod, n_max=n_max)

        self.energy_threshold = physics_params.get("energy_threshold", 0.0)
        self.energy_bin_width = physics_params.get("energy_bin_width", 0.0)
        self.energy_max_factor = physics_params.get("energy_max_factor", 1.2)

        self.q_max = numerics_params.get("q_max", 1.0)
        self.log_wavelet = numerics_params.get("log_wavelet", True)
        self.q_min = numerics_params.get(
            "q_min", self.energy_threshold / (const.VESC + const.VE)
        )
        self.eps = self.q_min / self.q_max

        # A bare construction (no params) is a container for import_hdf5,
        # which overwrites all attributes; only validate eps when projection
        # parameters were actually supplied.
        if (physics_params or numerics_params) and self.log_wavelet:
            if not (0.0 < self.eps < 1.0):
                raise ValueError(
                    "FormFactor: log_wavelet=True requires 0 < q_min/q_max < 1, "
                    f"but eps = q_min/q_max = {self.eps:.3e}. Set a positive "
                    "'energy_threshold' in physics_params or an explicit 'q_min' "
                    "in numerics_params."
                )

        # Store additional info
        self.info["energy_threshold"] = self.energy_threshold
        self.info["energy_bin_width"] = self.energy_bin_width
        self.info["energy_max_factor"] = self.energy_max_factor
        self.info["q_max"] = self.q_max
        self.info["log_wavelet"] = self.log_wavelet

        if "model" in physics_params:
            self.info["model"] = physics_params["model"]

    def import_hdf5(self, filename: str, groupname: str, dataname: str = "data"):
        super().import_hdf5(filename, groupname, dataname)
        keys_to_load = [
            # "n_bins",
            # "l_max",
            # "l_mod",
            # "n_max",
            "q_max",
            "energy_threshold",
            "energy_bin_width",
            "log_wavelet",
        ]
        for key in keys_to_load:
            if key in self.info:
                if key == "log_wavelet":
                    self.log_wavelet = self.info[key] in [
                        True,
                        "True",
                        "true",
                        "1",
                        1,
                    ]
                else:
                    setattr(self, key, float(self.info[key]))
            else:
                raise KeyError(f"Key '{key}' not found in HDF5 info.")
        if self.log_wavelet:
            if "log_wavelet_eps" in self.info:
                self.eps = float(self.info["log_wavelet_eps"])
            else:
                raise KeyError("Key 'log_wavelet_eps' not found in HDF5 info.")

        return self

    def project(self, params: dict):
        """
        Project the material form factor onto basis functions by quadrature.

        Parameters
        ----------
        params : dict
            Projection parameters. Required keys:
                - n_grid: tuple of ints (n_a, n_b, n_c) for the projection grid.
                  n_a (radial) must be a power of two greater than n_max.
                - material_input: path to the material input file.
                - physics_model_input: path to the DM-SM coupling input file.
                - numerics_input: path to the numerics input file.
            Optional keys:
                - chunk_size: q points per phonopy batch.
                  Default: sized automatically for ~1.0 GiB memory usage.
        """

        model = self.info.get("model")
        logger.info(
            f"\n=== Material form factor projection"
            f"{f' -- {model}' if model else ''} ==="
        )
        start_total_time = time.time()

        # Phonon data
        phonon_file, phonopy_params, c_dict, n_DW_params = (
            phonopy_funcs.get_material_data(
                params["material_input"],
                params["physics_model_input"],
                params["numerics_input"],
            )
        )

        l_max = self.l_max
        l_mod = self.l_mod
        n_max = self.n_max

        energy_threshold = self.energy_threshold
        energy_bin_width = self.energy_bin_width
        energy_max_factor = self.energy_max_factor
        energy_max = phonopy_funcs.get_energy_max(phonon_file, factor=energy_max_factor)
        energy_bin_num = int((energy_max - energy_threshold) / energy_bin_width) + 1
        self.n_bins = energy_bin_num
        self.info["n_bins"] = self.n_bins

        q_max = self.q_max
        log_wavelet = self.log_wavelet
        eps = self.eps if log_wavelet else 0.0

        n_a, n_b, n_c = params["n_grid"]
        self.info["n_grid"] = params["n_grid"]
        _validate_radial_grid(n_a, n_max)

        chunk_size = params.get("chunk_size")

        if log_wavelet:
            logger.info(
                f"    Using log wavelet mesh for projection with eps = {eps:.2e}."
            )
        else:
            logger.info("    Standard (power) wavelet basis in q.")
        logger.info(f"    Reference momentum q_max = {q_max:.2e} eV.")
        logger.info(
            f"    Projecting onto basis with l_max={l_max}, "
            f"l_mod={l_mod}, n_max={n_max}."
        )
        logger.info(f"    Grid size: n_a={n_a}, n_b={n_b}, n_c={n_c}")
        logger.info(
            f"    Energy bins: {energy_bin_num} bins from "
            f"{energy_threshold:.2e} eV to {energy_max:.2e} eV"
        )
        logger.info("Generating grids and evaluating the material form factor...")
        start_time = time.time()

        # Prepare grid points and basis function values
        lm_list = np.array(
            [
                (ell, m)
                for ell in range(0, l_max + 1, l_mod)
                for m in range(-ell, ell + 1)
            ],
            dtype=np.int64,
        )
        if log_wavelet:
            self.info["log_wavelet_eps"] = eps
            q_xyz_list, y_lm_vals, jacob_vals = gen_mesh_ylm_jacob_log(
                lm_list=lm_list, u_max=q_max, n_a=n_a, n_b=n_b, n_c=n_c, eps=eps
            )
        else:
            eps = 0.0
            q_xyz_list, y_lm_vals, jacob_vals = gen_mesh_ylm_jacob(
                lm_list=lm_list, u_max=q_max, n_a=n_a, n_b=n_b, n_c=n_c
            )

        # Calculate material form factor on grid
        form_factor_bin_vals = phonopy_funcs.form_factor(
            q_xyz_list,
            energy_threshold,
            energy_bin_width,
            energy_max,
            n_DW_params,
            phonopy_params,
            c_dict,
            phonon_file,
            chunk_size=chunk_size,
        ).reshape(energy_bin_num, n_a, n_b, n_c)
        del q_xyz_list

        end_time = time.time()
        logger.info(
            f"    (material form factor evaluation took {end_time - start_time:.1f} s)"
        )
        logger.info("Projecting material form factor onto basis functions...")

        # Project material form factor onto basis functions
        for idx_bin in range(energy_bin_num):
            self.fnlms[idx_bin] = Fnlm(
                l_max=l_max, l_mod=self.l_mod, n_max=n_max, info=self.info
            )

            if idx_bin % (energy_bin_num // 5 + 1) == 0:
                logger.info(
                    f"    bin {idx_bin}/{energy_bin_num - 1}, "
                    f"{time.time() - start_total_time:.0f} s elapsed"
                )

            self.fnlms[idx_bin].f_lm_n = proj_get_f_lm_n(
                n_max=n_max,
                lm_list=lm_list,
                func_vals=form_factor_bin_vals[idx_bin, :, :, :],
                y_lm_vals=y_lm_vals,
                jacob_vals=jacob_vals,
                n_a=n_a,
                log_wavelet=log_wavelet,
                eps=eps,
            )

        end_total_time = time.time()
        logger.info(f"=== Done in {end_total_time - start_total_time:.1f} s ===")


class McalI:
    """
    McalI class.

    Parameters
    ----------
    physics_params : dict or None
        Physics parameters required for McalI projection.
            - fdm: tuple, Dark matter form factor parameters (a,b) with
                   F_DM(q,v) = (q/q0)**a * (v/v0)**b
            - energy: energy transfer.
            - mass_dm: dark matter mass.
        Optional keys:
            - q0_fdm: reference momentum for FDM grid. Default is Bohr momentum
            - mass_sm: standard model particle mass.

    numerics_params : dict or None
        Numerical parameters required for McalI projection.
            - l_max: maximum angular momentum quantum number.
            - nv_max: maximum velocity radial quantum number.
            - nq_max: maximum momentum radial quantum number.
            - v_max: reference velocity scale.
            - q_max: reference momentum scale.
        Optional keys:
            - l_mod: denotes parity of l values (1 for all, 2 for even only).
            - log_wavelet_q: whether to use log wavelet mesh for q. Default is True.
            - eps_q: epsilon parameter for log wavelet mesh.
    """

    def __init__(
        self,
        physics_params: dict | None = None,
        numerics_params: dict | None = None,
    ):
        if physics_params is None:
            physics_params = {}
        if numerics_params is None:
            numerics_params = {}

        self.l_max = numerics_params.get("l_max", -1)
        self.l_mod = numerics_params.get("l_mod", 1)
        self.nv_max = numerics_params.get("nv_max", -1)
        self.nq_max = numerics_params.get("nq_max", -1)
        self.v_max = numerics_params.get("v_max", 1.0)
        self.q_max = numerics_params.get("q_max", 1.0)

        self.fdm = physics_params.get("fdm", (0, 0))
        self.q0_fdm = physics_params.get("q0_fdm", const.Q_BOHR)
        self.energy = physics_params.get("energy", 0.0)
        self.mass_dm = physics_params.get("mass_dm", 1.0)
        self.mass_sm = physics_params.get("mass_sm", const.M_NUCL)

        self.log_wavelet_q = numerics_params.get("log_wavelet_q", True)
        self.eps_q = numerics_params.get("eps_q", 1.0)

        if self.l_mod not in [1, 2]:
            raise ValueError("l_mod must be either 1 (all l) or 2 (even l only).")

        # A bare construction (no params) is a container for import_hdf5,
        # which overwrites all attributes; only validate eps_q when projection
        # parameters were actually supplied.
        if (physics_params or numerics_params) and self.log_wavelet_q:
            if self.eps_q <= 0.0 or self.eps_q >= 1.0:
                raise ValueError("eps_q must be in (0, 1) for log wavelet basis in q.")

        self.kernel = np.zeros(
            (self.l_max // self.l_mod + 1, self.nv_max + 1, self.nq_max + 1),
            dtype=float,
        )
        self.info = {}

    def export_hdf5(
        self,
        filename,
        groupname,
        dataname="data",
        write_info=True,
        dtype: npt.DTypeLike = np.float64,
        log_info=True,
    ):
        if not filename.endswith(".hdf5"):
            filename += ".hdf5"
        with h5py.File(filename, "a") as h5f:
            grp = h5f.require_group(groupname)
            if dataname in grp:
                del grp[dataname]
            dset = grp.create_dataset(dataname, data=self.kernel, dtype=dtype)
            dset.attrs["format_version"] = HDF5_FORMAT_VERSION

            if write_info:
                dset.attrs["l_max"] = self.l_max
                dset.attrs["l_mod"] = self.l_mod
                dset.attrs["nv_max"] = self.nv_max
                dset.attrs["nq_max"] = self.nq_max
                dset.attrs["v_max"] = self.v_max
                dset.attrs["q_max"] = self.q_max

                dset.attrs["fdm"] = self.fdm
                dset.attrs["q0_fdm"] = self.q0_fdm
                dset.attrs["energy"] = self.energy
                dset.attrs["mass_dm"] = self.mass_dm
                dset.attrs["mass_sm"] = self.mass_sm

                dset.attrs["log_wavelet_q"] = self.log_wavelet_q
                dset.attrs["eps_q"] = self.eps_q

                for key, value in self.info.items():
                    dset.attrs[key] = value

        if log_info:
            logger.info(
                f"McalI data written to {filename} in group {groupname}/{dataname}."
            )

    def import_hdf5(
        self,
        filename: str,
        groupname: str,
        dataname: str = "data",
        log_info=True,
    ):
        if not filename.endswith(".hdf5"):
            filename += ".hdf5"
        with h5py.File(filename, "r") as h5f:
            grp = h5f[groupname]
            dset = grp[dataname]
            self.kernel = dset[()]

            keys_to_load = [
                "l_max",
                "l_mod",
                "nv_max",
                "nq_max",
                "v_max",
                "q_max",
                "fdm",
                "q0_fdm",
                "energy",
                "mass_dm",
                "mass_sm",
                "log_wavelet_q",
                "eps_q",
            ]
            for key in keys_to_load:
                if key in dset.attrs:
                    setattr(self, key, dset.attrs[key])
                else:
                    raise KeyError(f"Key '{key}' not found in HDF5 attributes.")

            self.info = {
                key: dset.attrs[key]
                for key in dset.attrs
                if key not in keys_to_load and key != "format_version"
            }

        if log_info:
            logger.info(
                f"McalI data read from {filename} in group {groupname}/{dataname}."
            )

        return self

    def project(self, log_info=True):
        # BinnedMcalI drives this once per energy bin; log_info=False demotes
        # the per-call chatter to debug so the binned run logs once, not per bin.
        log = logger.info if log_info else logger.debug

        log("\n=== Kinematic scattering matrix (single energy) ===")
        start_time = time.time()

        l_max = self.l_max
        l_mod = self.l_mod
        nv_max = self.nv_max
        nq_max = self.nq_max
        v_max = self.v_max
        q_max = self.q_max
        fdm = self.fdm
        q0_fdm = self.q0_fdm
        energy = self.energy
        mass_dm = self.mass_dm
        mass_sm = self.mass_sm
        log_wavelet_q = self.log_wavelet_q
        eps_q = self.eps_q

        log(
            f"    Using FDM form factor parameters: fdm={fdm}, "
            f"q0_fdm={q0_fdm}, v0_fdm={1.0}."
        )
        log(
            f"    Using DM and SM parameters: mass_dm={mass_dm:.2e} eV, "
            f"mass_sm={mass_sm:.2e} eV, energy={energy:.2e} eV."
        )
        if log_wavelet_q:
            log(f"    Log wavelet basis in q with eps_q={eps_q:.2e}.")
        else:
            log("    Standard (power) wavelet basis in q.")
        log(f"    Reference velocity v_max = {v_max:.2e}.")
        log(f"    Reference momentum q_max = {q_max:.2e} eV.")
        log(
            f"    Projecting onto basis with l_max={l_max}, l_mod={l_mod}, "
            f"nv_max={nv_max}, nq_max={nq_max}."
        )
        log("Calculating McalI matrix coefficients...")

        # Projection
        self.kernel = self._compute_kernel()

        end_time = time.time()
        log(f"=== Done in {end_time - start_time:.1f} s ===")

    def _compute_kernel(self):
        l_max = self.l_max
        l_mod = self.l_mod
        nv_max = self.nv_max
        nq_max = self.nq_max

        for name, val in (("l_max", l_max), ("nv_max", nv_max), ("nq_max", nq_max)):
            if not isinstance(val, (int, np.integer)) or val < 0:
                raise ValueError(
                    f"McalI: {name} must be a non-negative integer, got {val!r}."
                )
        if l_mod not in (1, 2):
            raise ValueError(
                f"McalI: l_mod must be 1 (all l) or 2 (even l only), got {l_mod!r}."
            )
        if not (isinstance(self.fdm, (tuple, list, np.ndarray)) and len(self.fdm) == 2):
            raise ValueError(
                f"McalI: fdm must be a length-2 (a, b) tuple, got {self.fdm!r}."
            )
        if self.log_wavelet_q and not (0.0 < self.eps_q < 1.0):
            raise ValueError(
                "McalI: eps_q must be in (0, 1) for the log-wavelet q "
                f"basis, got {self.eps_q!r}."
            )

        v_max = self.v_max
        q_max = self.q_max
        fdm = self.fdm
        q0_fdm = self.q0_fdm
        v0_fdm = 1.0  # reference velocity for FDM form factor
        energy = self.energy
        mass_dm = self.mass_dm
        mass_sm = self.mass_sm
        log_wavelet_q = self.log_wavelet_q
        eps_q = self.eps_q

        (a, b) = fdm
        q_star = np.sqrt(2 * mass_dm * energy)
        v_star = q_star / mass_dm
        mass_reduced_sq = (mass_dm * mass_sm) ** 2 / (mass_dm + mass_sm) ** 2
        factor = (
            q_max
            / v_max**5
            / (2 * mass_dm * mass_reduced_sq)
            * (2 * energy) ** 2
            * (q_star / q0_fdm) ** a
            * (v_star / v0_fdm) ** b
        )

        return analytic.kin_matrix(
            l_max,
            l_mod,
            nv_max,
            nq_max,
            v_max,
            q_max,
            log_wavelet_q,
            eps_q,
            a,
            b,
            q_star,
            v_star,
            factor,
        )


class BinnedMcalI:
    """
    Binned McalI class.

    Parameters
    ----------
    physics_params : dict or None
        Physics parameters required for Binned McalI projection.
            - fdm: tuple, Dark matter form factor parameters (a,b) with
                   F_DM(q,v) = (q/q0)**a * (v/v0)**b
            - energy_threshold: minimum energy transfer.
            - energy_bin_width: width of each energy bin.
            - mass_dm: dark matter mass.
        Optional keys:
            - q0_fdm: reference momentum for FDM grid. Default is Bohr momentum
            - mass_sm: standard model particle mass. Default is nucleon mass.

    numerics_params : dict or None
        Numerical parameters required for Binned McalI projection.
            - l_max: maximum angular momentum quantum number.
            - nv_max: maximum velocity radial quantum number.
            - nq_max: maximum momentum radial quantum number.
            - v_max: reference velocity scale.
            - q_max: reference momentum scale.
            - n_bins: number of energy bins.
        Optional keys:
            - l_mod: denotes parity of l values (1 for all, 2 for even only).
            - log_wavelet_q: whether to use log wavelet mesh for q. Default is True.
            - eps_q: epsilon parameter for log wavelet mesh.
    """

    def __init__(
        self,
        physics_params: dict | None = None,
        numerics_params: dict | None = None,
    ):
        if physics_params is None:
            physics_params = {}
        if numerics_params is None:
            numerics_params = {}

        self.l_max = numerics_params.get("l_max", -1)
        self.l_mod = numerics_params.get("l_mod", 1)
        self.nv_max = numerics_params.get("nv_max", -1)
        self.nq_max = numerics_params.get("nq_max", -1)
        self.v_max = numerics_params.get("v_max", 1.0)
        self.q_max = numerics_params.get("q_max", 1.0)

        self.fdm = physics_params.get("fdm", (0, 0))
        self.q0_fdm = physics_params.get("q0_fdm", const.Q_BOHR)
        self.n_bins = numerics_params.get("n_bins", 0)
        self.energy_threshold = physics_params.get("energy_threshold", 0.0)
        self.energy_bin_width = physics_params.get("energy_bin_width", 0.0)
        self.mass_dm = physics_params.get("mass_dm", 1.0)
        self.mass_sm = physics_params.get("mass_sm", const.M_NUCL)

        self.log_wavelet_q = numerics_params.get("log_wavelet_q", True)
        self.eps_q = numerics_params.get("eps_q", 1.0)

        if self.l_mod not in [1, 2]:
            raise ValueError("l_mod must be either 1 (all l) or 2 (even l only).")

        # A bare construction (no params) is a container for import_hdf5,
        # which overwrites all attributes; only validate eps_q when projection
        # parameters were actually supplied.
        if (physics_params or numerics_params) and self.log_wavelet_q:
            if self.eps_q <= 0.0 or self.eps_q >= 1.0:
                raise ValueError("eps_q must be in (0, 1) for log wavelet basis in q.")

        self.mcalIs = {}  # to be filled after projection
        self.info = {}

    def export_hdf5(
        self,
        filename,
        groupname,
        dataname="data",
        write_sub_info=True,
        dtype: npt.DTypeLike = np.float64,
    ):
        if not filename.endswith(".hdf5"):
            filename += ".hdf5"
        with h5py.File(filename, "a") as h5f:
            grp = h5f.require_group(groupname)

            grp.attrs["format_version"] = HDF5_FORMAT_VERSION
            grp.attrs["n_bins"] = self.n_bins
            grp.attrs["l_max"] = self.l_max
            grp.attrs["l_mod"] = self.l_mod
            grp.attrs["nv_max"] = self.nv_max
            grp.attrs["nq_max"] = self.nq_max
            grp.attrs["v_max"] = self.v_max
            grp.attrs["q_max"] = self.q_max

            grp.attrs["fdm"] = self.fdm
            grp.attrs["q0_fdm"] = self.q0_fdm
            grp.attrs["energy_threshold"] = self.energy_threshold
            grp.attrs["energy_bin_width"] = self.energy_bin_width
            grp.attrs["mass_dm"] = self.mass_dm
            grp.attrs["mass_sm"] = self.mass_sm

            grp.attrs["log_wavelet_q"] = self.log_wavelet_q
            grp.attrs["eps_q"] = self.eps_q

            for key, value in self.info.items():
                grp.attrs[key] = value

            for idx_bin, mcalI in self.mcalIs.items():
                grp.require_group(f"bin_{idx_bin}")
                mcalI.export_hdf5(
                    filename,
                    f"{groupname}/bin_{idx_bin}",
                    dataname,
                    write_info=write_sub_info,
                    dtype=dtype,
                    log_info=False,
                )

        logger.info(
            f"BinnedMcalI data written to {filename} in group {groupname}/bin_*."
        )

    def import_hdf5(self, filename: str, groupname: str, dataname: str = "data"):
        if not filename.endswith(".hdf5"):
            filename += ".hdf5"
        with h5py.File(filename, "r") as h5f:
            grp = h5f[groupname]

            keys_to_load = [
                "n_bins",
                "l_max",
                "l_mod",
                "nv_max",
                "nq_max",
                "v_max",
                "q_max",
                "fdm",
                "q0_fdm",
                "energy_threshold",
                "energy_bin_width",
                "mass_dm",
                "mass_sm",
                "log_wavelet_q",
                "eps_q",
            ]
            for key in keys_to_load:
                if key in grp.attrs:
                    setattr(self, key, grp.attrs[key])
                else:
                    raise KeyError(f"Key '{key}' not found in HDF5 attributes.")

            self.info = {
                key: grp.attrs[key]
                for key in grp.attrs
                if key not in keys_to_load and key != "format_version"
            }

            self.mcalIs = {}
            for idx_bin in range(self.n_bins):
                grp.require_group(f"bin_{idx_bin}")
                mcalI = McalI()
                mcalI.import_hdf5(
                    filename, f"{groupname}/bin_{idx_bin}", dataname, log_info=False
                )
                self.mcalIs[idx_bin] = mcalI

        self._check_consistency()

        logger.info(
            f"BinnedMcalI data read from {filename} in group {groupname}/bin_*."
        )

        return self

    def project(self):

        logger.info(
            f"\n=== Kinematic scattering matrix projection -- {self.n_bins} bins ==="
        )
        start_time = time.time()

        physics_params_keys = [
            "fdm",
            "q0_fdm",
            "mass_dm",
            "mass_sm",
        ]
        numerics_params_keys = [
            "l_max",
            "l_mod",
            "nv_max",
            "nq_max",
            "v_max",
            "q_max",
            "log_wavelet_q",
            "eps_q",
        ]
        physics_params = {key: self.__dict__[key] for key in physics_params_keys}
        numerics_params = {key: self.__dict__[key] for key in numerics_params_keys}

        logger.info(
            f"    Using scaling parameters: fdm={self.fdm}, "
            f"q0_fdm={self.q0_fdm:.2e} eV, v0_fdm={1.0}."
        )
        logger.info(
            f"    Using mass parameters: mass_dm={self.mass_dm:.2e} eV, "
            f"mass_sm={self.mass_sm:.2e} eV"
        )
        logger.info(
            f"    {self.n_bins} energy bins from "
            f"energy_threshold={self.energy_threshold:.2e} eV with "
            f"energy_bin_width={self.energy_bin_width:.2e} eV."
        )
        if self.log_wavelet_q:
            logger.info(f"    Log wavelet basis in q with eps_q={self.eps_q:.2e}.")
        else:
            logger.info("    Standard (linear) wavelet basis in q.")
        logger.info(f"    Reference velocity v_max = {self.v_max:.2e}.")
        logger.info(f"    Reference momentum q_max = {self.q_max:.2e} eV.")
        logger.info(
            f"    Projecting onto basis with l_max={self.l_max}, "
            f"nv_max={self.nv_max}, nq_max={self.nq_max}."
        )
        logger.info("Calculating McalI matrix coefficients...")

        n_bins = self.n_bins

        for idx_bin in range(n_bins):
            if idx_bin % (n_bins // 5 + 1) == 0:
                logger.info(
                    f"    bin {idx_bin}/{n_bins - 1}, "
                    f"{time.time() - start_time:.0f} s elapsed"
                )

            energy = self.energy_threshold + (idx_bin + 0.5) * self.energy_bin_width
            physics_params["energy"] = energy

            mcalI = McalI(physics_params, numerics_params)
            mcalI.project(log_info=False)
            self.mcalIs[idx_bin] = mcalI

        end_time = time.time()
        logger.info(f"=== Done in {end_time - start_time:.1f} s ===")

    def _check_consistency(self):
        for idx_bin, mcalI in self.mcalIs.items():
            if mcalI.l_max != self.l_max:
                raise ValueError(f"Inconsistent l_max in bin {idx_bin}.")
            if mcalI.l_mod != self.l_mod:
                raise ValueError(f"Inconsistent l_mod in bin {idx_bin}.")
            if mcalI.nv_max != self.nv_max:
                raise ValueError(f"Inconsistent nv_max in bin {idx_bin}.")
            if mcalI.nq_max != self.nq_max:
                raise ValueError(f"Inconsistent nq_max in bin {idx_bin}.")
            if mcalI.v_max != self.v_max:
                raise ValueError(f"Inconsistent v_max in bin {idx_bin}.")
            if mcalI.q_max != self.q_max:
                raise ValueError(f"Inconsistent q_max in bin {idx_bin}.")
            if tuple(mcalI.fdm) != tuple(self.fdm):
                raise ValueError(f"Inconsistent fdm in bin {idx_bin}.")
            if mcalI.q0_fdm != self.q0_fdm:
                raise ValueError(f"Inconsistent q0_fdm in bin {idx_bin}.")
            if mcalI.mass_dm != self.mass_dm:
                raise ValueError(f"Inconsistent mass_dm in bin {idx_bin}.")
            if mcalI.mass_sm != self.mass_sm:
                raise ValueError(f"Inconsistent mass_sm in bin {idx_bin}.")
            if mcalI.log_wavelet_q != self.log_wavelet_q:
                raise ValueError(f"Inconsistent log_wavelet_q in bin {idx_bin}.")
            if mcalI.eps_q != self.eps_q:
                raise ValueError(f"Inconsistent eps_q in bin {idx_bin}.")
