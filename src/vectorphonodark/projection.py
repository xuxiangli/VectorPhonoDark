import numpy as np
import numba
import time
import os
import csv
import h5py

import vsdm
import phonopy

from . import constants as const
from . import basis_funcs
from . import utility
from .utility import C_GREEN, C_CYAN, C_RESET
from . import phonopy_funcs
from . import physics
from . import analytic


@numba.njit
def generate_mesh_ylm_jacob(
    lm_list: list[tuple[int, int]],
    u_max: float,
    n_a: int,
    n_b: int,
    n_c: int,
    power_a: float = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate mesh points, spherical harmonic values, and Jacobian values on the mesh.

    Args:
        lm_list:
            list of tuples: List of (l, m) tuples.
        u_max:
            float: Maximum radial value.
        n_a:
            int: Number of radial grid points.
        n_b:
            int: Number of theta grid points.
        n_c:
            int: Number of phi grid points.
        power_a:
            float: Power for radial grid spacing.
    Returns:
        u_xyz_list: An array of shape (n_a*n_b*n_c, 3) representing
                    Cartesian coordinates of the mesh points.
        y_lm_vals: A dictionary with keys as (l, m) tuples and values as
                   arrays of shape (n_b, n_c) representing spherical
                   harmonic values.
        jacob_list: An array of shape (n_a,) representing Jacobian values
                    for integration.
    """

    dcostheta = 2.0 / n_b
    theta_list = np.arccos(-np.linspace(-1.0 + dcostheta / 2, 1.0 - dcostheta / 2, n_b))
    dphi = 2 * np.pi / n_c
    phi_list = np.linspace(dphi / 2, 2 * np.pi - dphi / 2, n_c)

    y_lm_vals = {}
    for l, m in lm_list:
        y_lm_vals[(l, m)] = np.array(
            [
                vsdm.ylm_real(l, m, theta, phi)
                for theta in theta_list
                for phi in phi_list
            ]
        ).reshape(n_b, n_c)

    da = 1.0 / n_a
    a_list = np.linspace(da / 2, 1.0 - da / 2, n_a)
    if power_a == 1:
        dr = da
        r_list = a_list
        jacob_vals = r_list**2 * dr * dcostheta * dphi
    else:
        dr_list = power_a * np.power(a_list, power_a - 1) * da
        r_list = np.power(a_list, power_a)
        jacob_vals = r_list**2 * dr_list * dcostheta * dphi

    u_sph_list = np.array(
        [
            [u_max * r, theta, phi]
            for r in r_list
            for theta in theta_list
            for phi in phi_list
        ]
    ).reshape(n_a * n_b * n_c, 3)
    u_xyz_list = utility.sph_to_cart(u_sph_list)

    return u_xyz_list, y_lm_vals, jacob_vals


@numba.njit
def generate_log_mesh_ylm_jacob(
    lm_list: list[tuple[int, int]],
    u_max: float,
    n_a: int,
    n_b: int,
    n_c: int,
    eps: float = 0.,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate log-spaced mesh points, spherical harmonic values, and Jacobian values on the mesh.

    Args:
        lm_list:
            list of tuples: List of (l, m) tuples.
        u_max:
            float: Maximum radial value.
        n_a:
            int: Number of radial grid points.
        n_b:
            int: Number of theta grid points.
        n_c:
            int: Number of phi grid points.
        eps:
            float: Minimum radial value (u_max = 1.0).
    Returns:
        u_xyz_list: An array of shape (n_a*n_b*n_c, 3) representing
                    Cartesian coordinates of the mesh points.
        y_lm_vals: A dictionary with keys as (l, m) tuples and values as
                   arrays of shape (n_b, n_c) representing spherical
                   harmonic values.
        jacob_list: An array of shape (n_a,) representing Jacobian values
                    for integration.
    """

    dcostheta = 2.0 / n_b
    theta_list = np.arccos(-np.linspace(-1.0 + dcostheta / 2, 1.0 - dcostheta / 2, n_b))
    dphi = 2 * np.pi / n_c
    phi_list = np.linspace(dphi / 2, 2 * np.pi - dphi / 2, n_c)

    y_lm_vals = {}
    for l, m in lm_list:
        y_lm_vals[(l, m)] = np.array(
            [
                vsdm.ylm_real(l, m, theta, phi)
                for theta in theta_list
                for phi in phi_list
            ]
        ).reshape(n_b, n_c)

    da = 1.0 / n_a
    a_list = np.linspace(da / 2, 1.0 - da / 2, n_a)

    length = np.log(1. / eps)
    r_list = eps * np.exp(a_list * length)
    jacob_list = r_list**2 * (r_list * length * da) * dcostheta * dphi

    u_sph_list = np.array(
        [
            [u_max * r, theta, phi]
            for r in r_list
            for theta in theta_list
            for phi in phi_list
        ]
    ).reshape(n_a * n_b * n_c, 3)
    u_xyz_list = utility.sph_to_cart(u_sph_list)

    return u_xyz_list, y_lm_vals, jacob_list


@numba.njit
def proj_get_wavelet_boundary(n, n_a, power_a=1) -> tuple[int, int, int]:
    """
    Get the boundary indices for integration based on wavelet order n.
    """

    if n == 0:
        r_min_idx = 0.
        r_mid_idx = n_a
        r_max_idx = n_a
    else:
        x_min, x_mid, x_max = basis_funcs.haar_support(n)
        # Linear or power-law grid
        if power_a == 1:
            r_min_idx = int(x_min * n_a)
            r_mid_idx = int(x_mid * n_a)
            r_max_idx = int(x_max * n_a)
        else:
            r_min_idx = int(np.power(x_min, 1.0 / power_a) * n_a)
            r_mid_idx = int(np.power(x_mid, 1.0 / power_a) * n_a)
            r_max_idx = int(np.power(x_max, 1.0 / power_a) * n_a)

    return r_min_idx, r_mid_idx, r_max_idx


@numba.njit
def proj_get_wavelet_boundary_log(n, n_a, eps) -> tuple[int, int, int]:
    """
    Get the boundary indices for integration based on log wavelet order n.
    """

    if n == 0:
        r_min_idx = 0.
        r_mid_idx = n_a
        r_max_idx = n_a
    else:
        x_min, x_mid, x_max = basis_funcs.haar_support_log(n, eps)
        length = np.log(1.0 / eps)
        r_min_idx = int(np.log(x_min / eps) / (length / n_a))
        r_mid_idx = int(np.log(x_mid / eps) / (length / n_a))
        r_max_idx = int(np.log(x_max / eps) / (length / n_a))

    return r_min_idx, r_mid_idx, r_max_idx


@numba.njit
def proj_integrate_3d(
    func_vals: np.ndarray,
    haar_vals: float,
    y_lm_vals: np.ndarray,
    jacob_vals: np.ndarray,
) -> float:
    """
    Perform 3D integration for projection using precomputed values.

    Args:
        func_vals:
            np.ndarray: The function values on the grid.
        haar_vals:
            float: The Haar basis function values in the region.
        y_lm_vals:
            np.ndarray: The spherical harmonic values on the grid.
        jacob_vals:
            np.ndarray: The Jacobian values on the grid.

    Returns:
        float: The result of the 3D integration.
    """
    n_r, n_theta, n_phi = func_vals.shape
    total = 0.0
    for i in range(n_r):
        temp = 0.0
        for j in range(n_theta):
            for k in range(n_phi):
                temp += func_vals[i, j, k] * y_lm_vals[j, k]
        total += temp * jacob_vals[i]
    return total * haar_vals


@numba.njit
def proj_get_f_nlm(
    n_list: np.ndarray,
    lm_list: list[tuple[int, int]],
    func_vals: np.ndarray,
    y_lm_vals: dict[tuple[int, int], np.ndarray],
    jacob_vals: np.ndarray,
    n_a: int,
    power_a: float = 1,
    log_wavelet: bool = False,
    eps: float = 1.,
    verbose: bool = False,
) -> dict[tuple[int, int, int], float]:
    """
    Project function values onto basis functions to obtain f_nlm coefficients.

    Args:
        n_list:
            np.ndarray: List of radial quantum numbers.
        lm_list:
            list[tuple[int, int]]: List of (l, m) tuples representing angular quantum numbers.
        func_vals:
            np.ndarray: The function values on the grid.
        y_lm_vals:
            dict[tuple[int, int], np.ndarray]: The spherical harmonic values on the grid.
        jacob_vals:
            np.ndarray: The Jacobian values on the grid.
        n_a:
            int: Number of radial points.
        power_a:
            float: Power parameter for radial scaling.
        verbose:
            bool: Whether to print verbose output.

    Returns:
        dict[tuple[int, int, int], float]: A dictionary with keys as (n, l, m) tuples and values as the corresponding f_nlm coefficients.
    """

    f_nlm = {}

    for n in n_list:

        # Get wavelet values and boundaries
        if log_wavelet:
            value = basis_funcs.haar_value_log(n, eps=eps, p=2)
            r_min_idx, r_mid_idx, r_max_idx = proj_get_wavelet_boundary_log(
                n, n_a, eps=eps
            )   
        else:
            value = basis_funcs.haar_value(n, dim=3)
            r_min_idx, r_mid_idx, r_max_idx = proj_get_wavelet_boundary(
                n, n_a, power_a=power_a
            )

        # Perform integration
        if n == 0:
            for l, m in lm_list:
                f_nlm[(n, l, m)] = proj_integrate_3d(
                    func_vals[r_min_idx:r_max_idx, :, :],
                    value[0],
                    y_lm_vals[(l, m)],
                    jacob_vals[r_min_idx:r_max_idx],
                )
        else:
            for l, m in lm_list:
                f_nlm[(n, l, m)] = proj_integrate_3d(
                    func_vals[r_min_idx:r_mid_idx, :, :],
                    value[0],
                    y_lm_vals[(l, m)],
                    jacob_vals[r_min_idx:r_mid_idx],
                )
                f_nlm[(n, l, m)] += proj_integrate_3d(
                    func_vals[r_mid_idx:r_max_idx, :, :],
                    value[1],
                    y_lm_vals[(l, m)],
                    jacob_vals[r_mid_idx:r_max_idx],
                )

    return f_nlm


class Fnlm:
    def __init__(self):
        self.l_max = -1
        self.l_mod = 1
        self.n_list = np.array([])
        self.f_lm_n = np.array([])
        self.info = {}

    def get_lm_index(self, l, m):
        if abs(m) > l or l > self.l_max or l % self.l_mod != 0:
            raise ValueError("Invalid (l, m) values.")
        return l**2 + l + m

    def get_n_index(self, n):
        if n not in self.n_list:
            raise ValueError("n value not in n_list.")
        if not hasattr(self, "n_index_map") or n not in self.n_index_map:
            self.n_index_map = {val: idx for idx, val in enumerate(self.n_list)}
        return self.n_index_map[n]

    def export_csv(self, filename, write_info=True, verbose=True):
        if not filename.endswith(".csv"):
            filename += ".csv"
        makeHeader = not os.path.exists(filename)
        with open(filename, mode="a") as file:
            writer = csv.writer(file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
            if makeHeader:
                if write_info:
                    bparams = [r"#"] + [
                        str(key) + ":" + str(value) for key, value in self.info.items()
                    ]
                    writer.writerow(bparams)
                header = [r"#n", "l", "m", "f_lm_n"]
                writer.writerow(header)
            for l in range(self.l_max + 1):
                for m in range(-l, l + 1):
                    idx_lm = self.get_lm_index(l, m)
                    for idx_n, n in enumerate(self.n_list):
                        row = [n, l, m, self.f_lm_n[idx_lm, idx_n]]
                        writer.writerow(row)

        if verbose:
            print(f"    Fnlm data written to {C_GREEN}{filename}{C_RESET}.")

    def import_csv(self, filename, verbose=True):
        if not filename.endswith(".csv"):
            filename += ".csv"
        # read header and get l_max, n_list, l_mod first
        with open(filename, mode="r") as file:
            reader = csv.reader(file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
            data = {}
            n_list = set()
            l_list = set()
            for row in reader:
                if row[0].startswith("#"):
                    if len(row) > 1 and ":" in row[1]:
                        for item in row[1:]:
                            key, value = item.split(":", 1)
                            if key == "l_max":
                                self.l_max = int(value)
                            elif key == "l_mod":
                                self.l_mod = int(value)
                            elif key == "n_list":
                                n_items = value.strip("[]").split(",")
                                self.n_list = np.array([int(n) for n in n_items])
                            else:
                                self.info[key] = value
                    continue
                else:
                    n, l, m = int(row[0]), int(row[1]), int(row[2])
                    value = float(row[3])
                    n_list.add(n)
                    l_list.add(l)
                    data[(n, l, m)] = value

            if self.l_max == -1:
                self.l_max = max(l_list)
            if self.l_mod == 1:
                self.l_mod = 2 if all(l % 2 == 0 for l in l_list) else 1
            if len(self.n_list) == 0:
                self.n_list = np.array(sorted(n_list))

            self.f_lm_n = np.zeros(
                (self.get_lm_index(self.l_max, self.l_max) + 1, len(self.n_list)),
                dtype=float,
            )
            for (n, l, m), value in data.items():
                idx_lm = self.get_lm_index(l, m)
                idx_n = self.get_n_index(n)
                self.f_lm_n[idx_lm, idx_n] = value

        if verbose:
            print(f"    Fnlm data read from {C_GREEN}{filename}{C_RESET}.")

    def export_hdf5(
        self, filename, groupname, dataname="data", write_info=True, verbose=True
    ):
        if not filename.endswith(".hdf5"):
            filename += ".hdf5"
        with h5py.File(filename, "a") as h5f:
            grp = h5f.require_group(groupname)
            if dataname in grp:
                del grp[dataname]
            dset = grp.create_dataset(dataname, data=self.f_lm_n)
            dset.attrs["l_max"] = self.l_max
            dset.attrs["l_mod"] = self.l_mod
            if "n_list" in grp:
                del grp["n_list"]
            grp.create_dataset("n_list", data=self.n_list)
            # dset.attrs["n_list"] = self.n_list
            if write_info:
                for key, value in self.info.items():
                    dset.attrs[key] = value
        if verbose:
            print(
                f"    Fnlm data written to {C_GREEN}{filename}{C_RESET}",
                f"in group {C_CYAN}{groupname}/{dataname}{C_RESET}.",
            )

    def import_hdf5(self, filename, groupname, dataname="data", verbose=True):
        if not filename.endswith(".hdf5"):
            filename += ".hdf5"
        with h5py.File(filename, "r") as h5f:
            grp = h5f[groupname]
            dset = grp[dataname]
            self.f_lm_n = dset[()]
            self.l_max = dset.attrs["l_max"]
            self.l_mod = dset.attrs["l_mod"]
            self.n_list = grp["n_list"][()]
            self.info = {
                key: dset.attrs[key]
                for key in dset.attrs
                if key not in ["l_max", "l_mod", "n_list"]
            }

        if verbose:
            print(
                f"    Fnlm data read from {C_GREEN}{filename}{C_RESET}",
                f"in group {C_CYAN}{groupname}/{dataname}{C_RESET}.",
            )


class BinnedFnlm:
    def __init__(self):
        self.n_bins = 0
        self.l_max = -1
        self.l_mod = 1
        self.n_list = np.array([])
        self.fnlms = {}
        self.info = {}

    def export_csv(self, filename, write_info=True, verbose=True):
        if filename.endswith(".csv"):
            filename = filename[:-4]
        for idx_bin in range(self.n_bins):
            bin_filename = filename + "_bin_" + str(idx_bin) + ".csv"
            self.fnlms[idx_bin].export_csv(
                bin_filename, write_info=write_info, verbose=False
            )
        if verbose:
            print(
                f"    BinnedFnlm data written to {C_GREEN}{filename}_bin_*.csv{C_RESET} files."
            )

    def import_csv(self, filename, verbose=True):
        if filename.endswith(".csv"):
            filename = filename[:-4]

        self.fnlms = {}
        idx_bin = 0
        while True:
            try:
                bin_filename = filename + "_bin_" + str(idx_bin) + ".csv"
                fnlm = Fnlm()
                fnlm.import_csv(bin_filename, verbose=False)
                self.fnlms[idx_bin] = fnlm
                self.info = fnlm.info
                idx_bin += 1
            except FileNotFoundError:
                break
        self.n_bins = idx_bin

        if self.n_bins == 0:
            raise ValueError("No bin files found for BinnedFnlm import.")

        # check consistency of l_max, n_list, l_mod
        l_max_list = [fnlm.l_max for fnlm in self.fnlms.values()]
        n_list_list = [fnlm.n_list for fnlm in self.fnlms.values()]
        l_mod_list = [fnlm.l_mod for fnlm in self.fnlms.values()]
        if not all(l_max == l_max_list[0] for l_max in l_max_list):
            raise ValueError("Inconsistent l_max among bins.")
        else:
            self.l_max = l_max_list[0]
        if not all(np.array_equal(n_list, n_list_list[0]) for n_list in n_list_list):
            raise ValueError("Inconsistent n_list among bins.")
        else:
            self.n_list = n_list_list[0]
        if not all(l_mod == l_mod_list[0] for l_mod in l_mod_list):
            raise ValueError("Inconsistent l_mod among bins.")
        else:
            self.l_mod = l_mod_list[0]

        if verbose:
            print(
                f"    BinnedFnlm data read from {C_GREEN}{filename}_bin_*.csv{C_RESET} files."
            )

    def export_hdf5(
        self, filename, groupname, dataname="data", write_info=True, verbose=True
    ):
        if not filename.endswith(".hdf5"):
            filename += ".hdf5"
        with h5py.File(filename, "a") as h5f:
            grp = h5f.require_group(groupname)
            grp.attrs["n_bins"] = self.n_bins
            grp.attrs["l_max"] = self.l_max
            if "n_list" in grp:
                del grp["n_list"]
            grp.create_dataset("n_list", data=self.n_list)
            # grp.attrs["n_list"] = self.n_list
            for key, value in self.info.items():
                grp.attrs[key] = value
            for idx_bin, fnlm in self.fnlms.items():
                bin_grp = grp.require_group(f"bin_{idx_bin}")
                fnlm.export_hdf5(
                    filename,
                    f"{groupname}/bin_{idx_bin}",
                    dataname,
                    write_info=write_info,
                    verbose=False,
                )

        if verbose:
            print(
                f"    BinnedFnlm data written to {C_GREEN}{filename}{C_RESET}",
                f"in group {C_CYAN}{groupname}/bin_*{C_RESET}.",
            )

    def import_hdf5(self, filename, groupname, dataname="data", verbose=True):
        if not filename.endswith(".hdf5"):
            filename += ".hdf5"
        with h5py.File(filename, "r") as h5f:
            grp = h5f[groupname]
            self.n_bins = grp.attrs["n_bins"]
            self.l_max = grp.attrs["l_max"]
            self.n_list = grp["n_list"][()]
            # self.n_list = grp.attrs["n_list"]
            self.info = {
                key: grp.attrs[key]
                for key in grp.attrs
                if key not in ["n_bins", "l_max", "n_list"]
            }
            self.fnlms = {}
            for idx_bin in range(self.n_bins):
                bin_grp = grp[f"bin_{idx_bin}"]
                fnlm = Fnlm()
                fnlm.import_hdf5(
                    filename, f"{groupname}/bin_{idx_bin}", dataname, verbose=False
                )
                self.fnlms[idx_bin] = fnlm

        if verbose:
            print(
                f"    BinnedFnlm data read from {C_GREEN}{filename}{C_RESET}",
                f"in group {C_CYAN}{groupname}bin_*{C_RESET}.",
            )


class VDF(Fnlm):
    """
    Velocity Distribution Function class.

    Args:
        physics_params:
            dict: Physics parameters required for VDF projection.
                - vdf: the velocity distribution function.
                - vdf_params: parameters for the velocity distribution function.
        numerics_params:
            dict: Numerical parameters required for VDF projection.
                - v_max: reference velocity scale.
                - l_max: maximum angular momentum quantum number.
                - n_list: list of radial quantum numbers.
                - n_grid: number of grid points for velocity discretization.
    """

    def __init__(self, physics_params, numerics_params):
        super().__init__()
        self.l_max = numerics_params.get("l_max", -1)
        self.l_mod = numerics_params.get("l_mod", 1)
        self.n_list = numerics_params.get("n_list", [])

        self.vdf = physics_params.get("vdf", None)
        self.vdf_params = physics_params.get("vdf_params", {})

        self.v_max = numerics_params.get("v_max", 1.0)
        self.n_grid = numerics_params.get("n_grid", (32, 25, 25))

        # Store additional info
        for key, value in self.vdf_params.items():
            self.info["vdf_param_" + key] = value
        self.info["v_max"] = self.v_max
        self.info["n_grid"] = self.n_grid

        if "model" in physics_params:
            self.info["model"] = physics_params["model"]

    def import_csv(self, filename, verbose=True):
        super().import_csv(filename, verbose=verbose)
        self.v_max = float(self.info.get("v_max", 1.0))
        self.info["v_max"] = self.v_max

    def import_hdf5(self, filename, groupname, dataname="data", verbose=True):
        super().import_hdf5(filename, groupname, dataname, verbose)
        self.v_max = float(self.info.get("v_max", 1.0))
        self.info["v_max"] = self.v_max

    def project(self, params, verbose=False):

        if verbose:
            print(
                "\n    Starting projection of velocity distribution function onto basis functions..."
            )
            start_total_time = time.time()

        v_max = params.get("v_max", self.v_max)
        vdf = params.get("vdf", self.vdf)
        vdf_params = params.get("vdf_params", self.vdf_params)

        l_max = params.get("l_max", self.l_max)
        n_list = params.get("n_list", self.n_list)
        n_list = np.array(n_list)

        n_a, n_b, n_c = params.get("n_grid", self.n_grid)
        # p_a, p_b, p_c = params.get("power_grid", (1, 1, 1))

        # assert (
        #     p_b == 1 and p_c == 1
        # ), "Currently only power 1 is supported for angular grids."

        if verbose:
            if "model" in self.info:
                print(f"    Using VDF model: {self.info['model']}")
            print(f"    Parameters for VDF: {vdf_params}")
            print(f"    Reference velocity v_max = {v_max:.2e}.")
            print(f"    Projecting onto basis with l_max={l_max}, n_max={max(n_list)}.")
            print(f"    Grid size: n_a={n_a}, n_b={n_b}, n_c={n_c}")
            # print(f"    Grid rescaled: power_a={p_a}, power_b={p_b}, power_c={p_c}")

            print("\n    Generate grids and calculating form factor...")
            start_time = time.time()

        # Prepare grid points and basis function values
        lm_list = [(l, m) for l in range(l_max + 1) for m in range(-l, l + 1)]
        v_xyz_list, _, y_lm_vals, jacob_vals = generate_mesh_ylm_jacob(
            lm_list, v_max, n_a, n_b, n_c  # , p_a, p_b, p_c
        )

        # Calculate vdf on grid
        vdf_vals = np.array([vdf(v_vec, **vdf_params) for v_vec in v_xyz_list]).reshape(
            n_a, n_b, n_c
        )
        del v_xyz_list

        if verbose:
            end_time = time.time()
            print(
                f"    VDF calculation completed in {end_time - start_time:.2f} seconds."
            )
            print("\n    Projecting VDF onto basis functions and saving results...")

        # Project vdf onto basis functions
        f_nlm = proj_get_f_nlm(
            n_list,
            lm_list,
            vdf_vals,
            y_lm_vals,
            jacob_vals,
            "haar",
            n_a,
            # p_a,
            verbose=verbose,
        )
        del vdf_vals, y_lm_vals, jacob_vals

        # Store results
        self.v_max = v_max
        self.info["v_max"] = self.v_max
        self.l_max = l_max
        self.n_list = n_list
        self.f_lm_n = np.zeros(
            (self.get_lm_index(l_max, l_max) + 1, len(n_list)), dtype=float
        )
        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                idx_lm = self.get_lm_index(l, m)
                for idx_n, n in enumerate(n_list):
                    self.f_lm_n[idx_lm, idx_n] = f_nlm.get((n, l, m), 0.0)

        if verbose:
            print("    Projection completed.")
            end_total_time = time.time()
            print(
                f"    Total projection time: {end_total_time - start_total_time:.2f} seconds."
            )


class FormFactor(BinnedFnlm):
    """
    Form Factor class.

    Args:
        physics_params:
            dict: Physics parameters required for Form Factor projection.
                - energy_threshold: minimum energy transfer.
                - energy_bin_width: width of each energy bin.
                - energy_max_factor: factor to determine maximum energy.
        numerics_params:
            dict: Numerical parameters required for Form Factor projection.
                - q_max: reference momentum scale.
                - l_max: maximum angular momentum quantum number.
                - n_list: list of radial quantum numbers.
                - n_grid: number of grid points for momentum discretization.
                - special_mesh: whether to use special mesh and wavelets.
                - q_cut: whether to compute q_cut from Debye-Waller factor.
    """

    def __init__(self, physics_params, numerics_params):
        super().__init__()
        self.l_max = numerics_params.get("l_max", -1)
        self.l_mod = numerics_params.get("l_mod", 1)
        self.n_list = numerics_params.get("n_list", [])

        self.energy_threshold = physics_params.get("energy_threshold", 1e-3)
        self.energy_bin_width = physics_params.get("energy_bin_width", 1e-3)
        self.energy_max_factor = physics_params.get("energy_max_factor", 4.0)

        self.q_max = numerics_params.get("q_max", 1.0)
        self.n_grid = numerics_params.get("n_grid", (32, 25, 25))
        self.special_mesh = numerics_params.get("special_mesh", False)
        self.log_wavelet = numerics_params.get("log_wavelet", False)
        self.q_cut = numerics_params.get("q_cut", False)

        # Store additional info
        self.info["energy_threshold"] = self.energy_threshold
        self.info["energy_bin_width"] = self.energy_bin_width
        self.info["q_max"] = self.q_max
        self.info["n_grid"] = self.n_grid
        self.info["special_mesh"] = self.special_mesh
        self.info["log_wavelet"] = self.log_wavelet
        self.info["q_cut"] = self.q_cut

        if "model" in physics_params:
            self.info["model"] = physics_params["model"]

    def import_csv(self, filename, verbose=True):
        super().import_csv(filename, verbose=verbose)
        self.q_max = float(self.info.get("q_max", 1.0))
        self.info["q_max"] = self.q_max
        for fnlm in self.fnlms.values():
            fnlm.q_max = self.q_max
            fnlm.info["q_max"] = self.q_max

    def import_hdf5(self, filename, groupname, dataname="data", verbose=True):
        super().import_hdf5(filename, groupname, dataname, verbose=verbose)
        self.q_max = float(self.info.get("q_max", 1.0))
        self.info["q_max"] = self.q_max
        for fnlm in self.fnlms.values():
            fnlm.q_max = self.q_max
            fnlm.info["q_max"] = self.q_max

    def project(self, params, verbose=False):

        if verbose:
            print("\n    Starting projection of form factor onto basis functions...")
            start_total_time = time.time()

        # Phonon data
        phonon_file, phonopy_params, c_dict, n_DW_params = self._get_phonon_data(params)

        l_max = params.get("l_max", self.l_max)
        n_list = params.get("n_list", self.n_list)
        n_list = np.array(n_list)

        n_a, n_b, n_c = params.get("n_grid", self.n_grid)
        # p_a, p_b, p_c = params.get("power_grid", (1, 1, 1))
        log_wavelet = params.get("log_wavelet", self.log_wavelet)

        energy_threshold = params.get("energy_threshold", self.energy_threshold)
        energy_bin_width = params.get("energy_bin_width", self.energy_bin_width)
        energy_max_factor = params.get("energy_max_factor", self.energy_max_factor)
        energy_max = physics.get_energy_max(phonon_file, factor=energy_max_factor)
        energy_bin_num = int((energy_max - energy_threshold) / energy_bin_width) + 1

        q_max = params.get("q_max", self.q_max)
        q_cut_option = params.get("q_cut", self.q_cut)
        q_max = physics.get_q_max(
            q_max=q_max,
            q_cut_option=q_cut_option,
            phonon_file=phonon_file,
            atom_masses=phonopy_params["atom_masses"],
            verbose=verbose,
        )

        # Store results
        self.l_max = l_max
        self.n_list = n_list
        self.q_max = q_max
        self.log_wavelet = log_wavelet
        self.n_bins = energy_bin_num
        self.info["n_bins"] = self.n_bins

        if verbose:
            if "material" in self.info:
                print(f"    Material: {self.info['material']}")
            if log_wavelet:
                print("    Using log wavelet mesh for projection.")
            print(f"    Reference momentum q_max = {q_max:.2e} eV.")
            print(f"    Projecting onto basis with l_max={l_max}, n_max={max(n_list)}.")
            print(f"    Grid size: n_a={n_a}, n_b={n_b}, n_c={n_c}")
            # print(f"    Grid rescaled: power_a={p_a}, power_b={p_b}, power_c={p_c}")
            print(
                f"    Energy bins: {energy_bin_num} bins from {energy_threshold:.2e} eV to {energy_max:.2e} eV"
            )

            print("\n    Generate grids and calculating form factor...")
            start_time = time.time()

        # Prepare grid points and basis function values
        lm_list = [(l, m) for l in range(l_max + 1) for m in range(-l, l + 1)]
        if log_wavelet:
            q_min = energy_threshold / (const.VESC + const.VE)
            eps = q_min / q_max
            self.info["log_wavelet_eps"] = eps
            q_xyz_list, y_lm_vals, jacob_vals = generate_log_mesh_ylm_jacob(
                lm_list, q_max, n_a, n_b, n_c, eps=eps
            )
        else:
            eps = 0.
            q_xyz_list, y_lm_vals, jacob_vals = generate_mesh_ylm_jacob(
                lm_list, q_max, n_a, n_b, n_c
            )

        # Calculate form factor on grid
        form_factor_bin_vals = physics.form_factor(
            q_xyz_list,
            energy_threshold,
            energy_bin_width,
            energy_max,
            n_DW_params,
            phonopy_params,
            c_dict,
            phonon_file,
        ).reshape(energy_bin_num, n_a, n_b, n_c)
        del q_xyz_list

        if verbose:
            end_time = time.time()
            print(
                f"    Form factor calculation completed in {end_time - start_time:.2f} seconds."
            )
            print(
                "\n    Projecting form factor onto basis functions and saving results..."
            )

        # Project form factor onto basis functions
        for i_bin in range(energy_bin_num):

            self.fnlms[i_bin] = Fnlm()
            self.fnlms[i_bin].info = self.info

            if verbose:
                if i_bin % (energy_bin_num // 5 + 1) == 0:
                    print(f"      Projecting energy bin {i_bin}/{energy_bin_num-1}...")

            f_nlm = proj_get_f_nlm(
                n_list, lm_list,
                form_factor_bin_vals[i_bin, :, :, :], y_lm_vals, jacob_vals,
                n_a,
                # p_a,
                log_wavelet=log_wavelet, eps=eps, verbose=verbose,
            )
            self.fnlms[i_bin].l_max = l_max
            self.fnlms[i_bin].n_list = n_list

            self.fnlms[i_bin].f_lm_n = np.zeros(
                (
                    self.fnlms[i_bin].get_lm_index(l_max, l_max) + 1,
                    len(self.fnlms[i_bin].n_list),
                ),
                dtype=float,
            )
            for l in range(l_max + 1):
                for m in range(-l, l + 1):
                    idx_lm = self.fnlms[i_bin].get_lm_index(l, m)
                    for idx_n, n in enumerate(n_list):
                        self.fnlms[i_bin].f_lm_n[idx_lm, idx_n] = (
                            f_nlm.get((n, l, m), 0.0)
                        )

        if verbose:
            print("    Projection completed.")
            end_total_time = time.time()
            print(
                f"    Total projection time: {end_total_time - start_total_time:.2f} seconds."
            )

    @staticmethod
    def _get_phonon_data(params):

        material_input = params["material_input"]
        physics_model_input = params["physics_model_input"]
        numerics_input = params["numerics_input"]

        mat_input_mod_name = os.path.splitext(os.path.basename(material_input))[0]
        phys_input_mod_name = os.path.splitext(os.path.basename(physics_model_input))[0]
        num_input_mod_name = os.path.splitext(os.path.basename(numerics_input))[0]

        mat_mod = utility.import_file(mat_input_mod_name, os.path.join(material_input))
        phys_mod = utility.import_file(
            phys_input_mod_name, os.path.join(physics_model_input)
        )
        num_mod = utility.import_file(num_input_mod_name, os.path.join(numerics_input))

        material = mat_mod.material
        c_dict = phys_mod.c_dict

        poscar_path = os.path.join(os.path.split(material_input)[0], "POSCAR")
        force_sets_path = os.path.join(os.path.split(material_input)[0], "FORCE_SETS")
        born_path = os.path.join(os.path.split(material_input)[0], "BORN")

        if os.path.exists(born_path):
            born_exists = True
        else:
            print(
                "  There is no BORN file for "
                + material
                + ". PHONOPY calculations will process with .NAC. = FALSE\n"
            )
            born_exists = False

        if born_exists:
            phonon_file = phonopy.load(
                supercell_matrix=mat_mod.mat_properties_dict["supercell_dim"],
                primitive_matrix="auto",
                unitcell_filename=poscar_path,
                force_sets_filename=force_sets_path,
                is_nac=True,
                born_filename=born_path,
            )
        else:
            phonon_file = phonopy.load(
                supercell_matrix=mat_mod.mat_properties_dict["supercell_dim"],
                primitive_matrix="auto",
                unitcell_filename=poscar_path,
                force_sets_filename=force_sets_path,
            )

        phonopy_params = phonopy_funcs.get_phonon_file_data(phonon_file, born_exists)

        n_DW_params = {
            "n_DW_x": num_mod.numerics_parameters["n_DW_x"],
            "n_DW_y": num_mod.numerics_parameters["n_DW_y"],
            "n_DW_z": num_mod.numerics_parameters["n_DW_z"],
        }

        return phonon_file, phonopy_params, c_dict, n_DW_params


class McalI:
    def __init__(self, physics_params, numerics_params):
        self.l_max = numerics_params.get("l_max", -1)
        self.l_mod = numerics_params.get("l_mod", 1)
        self.nv_list = numerics_params.get("nv_list", [])
        self.nq_list = numerics_params.get("nq_list", [])
        self.v_max = numerics_params.get("v_max", 1.0)
        self.q_max = numerics_params.get("q_max", 1.0)

        self.fdm = physics_params.get("fdm", (0, 0))
        self.q0_fdm = physics_params.get("q0_fdm", const.Q_BOHR)
        self.energy = physics_params.get("energy", 0.0)
        self.mass_dm = physics_params.get("mass_dm", 10**6)
        self.mass_sm = physics_params.get("mass_sm", const.M_NUCL)

        self.log_wavelet_q = numerics_params.get("log_wavelet_q", False)
        self.eps_q = numerics_params.get("eps_q", 0.0)

        self.mcalI = np.array(
            (self.l_max + 1, len(self.nv_list), len(self.nq_list)), dtype=float
        )
        self.info = {}

    def export_hdf5(
        self, filename, groupname, dataname="data", write_info=True, verbose=True
    ):
        if not filename.endswith(".hdf5"):
            filename += ".hdf5"
        with h5py.File(filename, "a") as h5f:
            grp = h5f.require_group(groupname)
            if dataname in grp:
                del grp[dataname]
            dset = grp.create_dataset(dataname, data=self.mcalI)
            if write_info:
                dset.attrs["l_max"] = self.l_max
                if "nv_list" in grp:
                    del grp["nv_list"]
                grp.create_dataset("nv_list", data=self.nv_list)
                if "nq_list" in grp:
                    del grp["nq_list"]
                grp.create_dataset("nq_list", data=self.nq_list)
                # dset.attrs["nv_list"] = self.nv_list
                # dset.attrs["nq_list"] = self.nq_list
                dset.attrs["v_max"] = self.v_max
                dset.attrs["q_max"] = self.q_max
                dset.attrs["fdm"] = self.fdm
                dset.attrs["energy"] = self.energy
                dset.attrs["mass_dm"] = self.mass_dm
                dset.attrs["mass_sm"] = self.mass_sm
                dset.attrs["log_wavelet_q"] = self.log_wavelet_q
                dset.attrs["eps_q"] = self.eps_q
                for key, value in self.info.items():
                    dset.attrs[key] = value

        if verbose:
            print(
                f"    McalI data written to {C_GREEN}{filename}{C_RESET}",
                f"in group {C_CYAN}{groupname}/{dataname}{C_RESET}.",
            )

    def import_hdf5(self, filename, groupname, dataname="data", verbose=True):
        if not filename.endswith(".hdf5"):
            filename += ".hdf5"
        with h5py.File(filename, "r") as h5f:
            grp = h5f[groupname]
            dset = grp[dataname]
            self.mcalI = dset[()]

            self.nv_list = grp["nv_list"][()]
            self.nq_list = grp["nq_list"][()]

            keys_to_load = [
                "l_max",
                # "nv_list",
                # "nq_list",
                "v_max",
                "q_max",
                "fdm",
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
                key: dset.attrs[key] for key in dset.attrs if key not in keys_to_load
            }

        if verbose:
            print(
                f"    McalI data read from {C_GREEN}{filename}{C_RESET}",
                f"in group {C_CYAN}{groupname}/{dataname}{C_RESET}.",
            )

    def project(self, params, verbose=False):

        if verbose:
            print("\n    Starting projection of McalI onto basis functions...")
            start_time = time.time()

        l_max = params.get("l_max", self.l_max)
        nv_list = params.get("nv_list", self.nv_list)
        nq_list = params.get("nq_list", self.nq_list)
        v_max = params.get("v_max", self.v_max)
        q_max = params.get("q_max", self.q_max)
        fdm = params.get("fdm", self.fdm)
        q0_fdm = params.get("q0_fdm", self.q0_fdm)
        energy = params.get("energy", self.energy)
        mass_dm = params.get("mass_dm", self.mass_dm)
        mass_sm = params.get("mass_sm", self.mass_sm)
        log_wavelet_q = params.get("log_wavelet_q", self.log_wavelet_q)
        eps_q = params.get("eps_q", self.eps_q)
        if log_wavelet_q and (eps_q <= 0.0 or eps_q >= 1.0):
            raise ValueError("eps_q must be in (0, 1) for log wavelet basis in q.")
        
        if verbose:
            print(f"    Using FDM form factor parameters: fdm={fdm}, q0_fdm={q0_fdm}, v0_fdm={1.0}.")
            print(f"    Using DM and SM parameters: mass_dm={mass_dm:.2e} eV, mass_sm={mass_sm:.2e} eV, energy={energy:.2e} eV.")
            print(f"    Parameters for wavelets:")
            if log_wavelet_q:
                print(f"      Log wavelet basis in q with eps_q={eps_q:.2e}.")
            else:
                print(f"      Standard (linear) wavelet basis in q.")
            print(f"    Reference velocity v_max = {v_max:.2e}.")
            print(f"    Reference momentum q_max = {q_max:.2e} eV.")
            print(f"    Projecting onto basis with l_max={l_max}, nv_max={max(nv_list)}, nq_max={max(nq_list)}.")

            print("\n    Calculating McalI matrix coefficients...")

        # Store results
        self.l_max = l_max
        self.nv_list = nv_list
        self.nq_list = nq_list
        self.v_max = v_max
        self.q_max = q_max
        self.fdm = fdm
        self.q0_fdm = q0_fdm
        self.energy = energy
        self.mass_dm = mass_dm
        self.mass_sm = mass_sm
        self.log_wavelet_q = log_wavelet_q
        self.eps_q = eps_q

        # Projection
        self.mcalI = np.zeros((l_max + 1, len(nv_list), len(nq_list)), dtype=float)
        for l in range(l_max + 1):
            for idx_nv, nv in enumerate(nv_list):
                for idx_nq, nq in enumerate(nq_list):
                    self.mcalI[l, idx_nv, idx_nq] = self.getI_lvq_analytic((l, nv, nq))

        if verbose:
            print("    Projection completed.")
            end_time = time.time()
            print(f"    Total projection time: {end_time - start_time:.2f} seconds.")

    def getI_lvq_analytic(self, lnvq, verbose=False):
        """Analytic calculation for I(ell) matrix coefficients.

        Only available for 'tophat' and 'wavelet' bases (so far).

        Arguments:
            lnvq = (ell, nv, nq)
            verbose: whether to print output
        """
        v_max = self.v_max
        q_max = self.q_max
        mass_dm = self.mass_dm
        fdm = self.fdm  # DM-SM scattering form factor index
        # (a, b) = fdm
        q0_fdm = self.q0_fdm  # reference momentum for FDM form factor
        v0_fdm = 1.0  # reference velocity for FDM form factor
        mass_sm = self.mass_sm  # SM particle mass (mElec)
        energy = self.energy  # DM -> SM energy transfer
        # mass_reduced = (mass_dm * mass_sm) / (mass_dm + mass_sm)
        log_wavelet_q = self.log_wavelet_q
        eps_q = self.eps_q

        Ilvq = analytic.ilvq_analytic(
            lnvq, v_max, q_max, log_wavelet_q, eps_q,
            fdm, q0_fdm, v0_fdm, 
            mass_dm, mass_sm, energy, verbose=verbose
        )

        # Ilvq = analytic.ilvq_vsdm(
        #     lnvq, v_max, q_max, log_wavelet_q, eps_q,
        #     fdm, q0_fdm, v0_fdm, 
        #     mass_dm, mass_sm, energy, verbose=verbose
        # )

        return Ilvq


class BinnedMcalI:
    def __init__(self, physics_params, numerics_params):
        self.n_bins = numerics_params.get("n_bins", 0)
        self.l_max = numerics_params.get("l_max", -1)
        self.l_mod = numerics_params.get("l_mod", 1)
        self.nv_list = numerics_params.get("nv_list", [])
        self.nq_list = numerics_params.get("nq_list", [])
        self.v_max = numerics_params.get("v_max", 1.0)
        self.q_max = numerics_params.get("q_max", 1.0)

        self.fdm = physics_params.get("fdm", (0, 0))
        self.q0_fdm = physics_params.get("q0_fdm", const.Q_BOHR)
        self.energy_threshold = physics_params.get("energy_threshold", 1e-3)
        self.energy_bin_width = physics_params.get("energy_bin_width", 1e-3)
        self.mass_dm = physics_params.get("mass_dm", 10**6)
        self.mass_sm = physics_params.get("mass_sm", const.M_NUCL)

        self.log_wavelet_q = numerics_params.get("log_wavelet_q", False)
        self.eps_q = numerics_params.get("eps_q", 0.0)

        self.mcalIs = {}  # to be filled after projection
        self.info = {}  # to be filled with relevant info

    def export_hdf5(
        self, filename, groupname, dataname="data", write_sub_info=True, verbose=True
    ):
        if not filename.endswith(".hdf5"):
            filename += ".hdf5"
        with h5py.File(filename, "a") as h5f:
            grp = h5f.require_group(groupname)

            grp.attrs["n_bins"] = self.n_bins
            grp.attrs["l_max"] = self.l_max
            if "nv_list" in grp:
                del grp["nv_list"]
            grp.create_dataset("nv_list", data=self.nv_list)
            if "nq_list" in grp:
                del grp["nq_list"]
            grp.create_dataset("nq_list", data=self.nq_list)
            # grp.attrs["nv_list"] = self.nv_list
            # grp.attrs["nq_list"] = self.nq_list
            grp.attrs["v_max"] = self.v_max
            grp.attrs["q_max"] = self.q_max

            grp.attrs["fdm"] = self.fdm
            grp.attrs["energy_threshold"] = self.energy_threshold
            grp.attrs["energy_bin_width"] = self.energy_bin_width
            grp.attrs["mass_dm"] = self.mass_dm
            grp.attrs["mass_sm"] = self.mass_sm
            grp.attrs["log_wavelet_q"] = self.log_wavelet_q
            grp.attrs["eps_q"] = self.eps_q

            for key, value in self.info.items():
                grp.attrs[key] = value
            for idx_bin, mcalI in self.mcalIs.items():
                bin_grp = grp.require_group(f"bin_{idx_bin}")
                mcalI.export_hdf5(
                    filename,
                    f"{groupname}/bin_{idx_bin}",
                    dataname,
                    write_info=write_sub_info,
                    verbose=False,
                )

        if verbose:
            print(
                f"    BinnedMcalI data written to {C_GREEN}{filename}{C_RESET}",
                f"in group {C_CYAN}{groupname}/bin_*{C_RESET}.",
            )

    def import_hdf5(self, filename, groupname, dataname="data", verbose=True):
        if not filename.endswith(".hdf5"):
            filename += ".hdf5"
        with h5py.File(filename, "r") as h5f:
            grp = h5f[groupname]

            self.nv_list = grp["nv_list"][()]
            self.nq_list = grp["nq_list"][()]

            keys_to_load = [
                "n_bins",
                "l_max",
                # "nv_list",
                # "nq_list",
                "v_max",
                "q_max",
                "fdm",
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
                key: grp.attrs[key] for key in grp.attrs if key not in keys_to_load
            }

            self.mcalIs = {}
            for idx_bin in range(self.n_bins):
                bin_grp = grp[f"bin_{idx_bin}"]
                mcalI = McalI(physics_params={}, numerics_params={})
                mcalI.import_hdf5(
                    filename, f"{groupname}/bin_{idx_bin}", dataname, verbose=False
                )
                self.mcalIs[idx_bin] = mcalI

            # check consistency
            for idx_bin, mcalI in self.mcalIs.items():
                if mcalI.l_max != self.l_max:
                    raise ValueError(f"Inconsistent l_max in bin {idx_bin}.")
                if not np.array_equal(mcalI.nv_list, self.nv_list):
                    raise ValueError(f"Inconsistent nv_list in bin {idx_bin}.")
                if not np.array_equal(mcalI.nq_list, self.nq_list):
                    raise ValueError(f"Inconsistent nq_list in bin {idx_bin}.")
                if mcalI.v_max != self.v_max:
                    raise ValueError(f"Inconsistent v_max in bin {idx_bin}.")
                if mcalI.q_max != self.q_max:
                    raise ValueError(f"Inconsistent q_max in bin {idx_bin}.")
                if mcalI.log_wavelet_q != self.log_wavelet_q:
                    raise ValueError(f"Inconsistent log_wavelet_q in bin {idx_bin}.")
                if mcalI.eps_q != self.eps_q:
                    raise ValueError(f"Inconsistent eps_q in bin {idx_bin}.")
        if verbose:
            print(
                f"    BinnedMcalI data read from {C_GREEN}{filename}{C_RESET}",
                f"in group {C_CYAN}{groupname}bin_*{C_RESET}.",
            )

    def project(self, params, verbose=False):

        if verbose:
            print("\n    Starting projection of McalI onto basis functions...")
            start_time = time.time()

        params_copy = params.copy()

        n_bins = params_copy.get("n_bins", self.n_bins)

        if "l_max" not in params_copy:
            params_copy["l_max"] = self.l_max
        if "nv_list" not in params_copy:
            params_copy["nv_list"] = self.nv_list
        if "nq_list" not in params_copy:
            params_copy["nq_list"] = self.nq_list
        if "v_max" not in params_copy:
            params_copy["v_max"] = self.v_max
        if "q_max" not in params_copy:
            params_copy["q_max"] = self.q_max

        if "fdm" not in params_copy:
            params_copy["fdm"] = self.fdm
        if "q0_fdm" not in params_copy:
            params_copy["q0_fdm"] = self.q0_fdm
        if "energy_threshold" not in params_copy:
            params_copy["energy_threshold"] = self.energy_threshold
        if "energy_bin_width" not in params_copy:
            params_copy["energy_bin_width"] = self.energy_bin_width
        if "mass_dm" not in params_copy:
            params_copy["mass_dm"] = self.mass_dm
        if "mass_sm" not in params_copy:
            params_copy["mass_sm"] = self.mass_sm
        if "log_wavelet_q" not in params_copy:
            params_copy["log_wavelet_q"] = self.log_wavelet_q
        if "eps_q" not in params_copy:
            params_copy["eps_q"] = self.eps_q

        if verbose:
            print(f"    Using FDM form factor parameters: fdm={params_copy["fdm"]}, q0_fdm={params_copy["q0_fdm"]:.2e} eV, v0_fdm={1.0}.")
            print(f"    Using DM and SM parameters: mass_dm={params_copy["mass_dm"]:.2e} eV, mass_sm={params_copy["mass_sm"]:.2e} eV")
            print(f"    {n_bins} energy bins starting from")
            print(f"    energy_threshold={params_copy['energy_threshold']:.2e} eV, energy_bin_width={params_copy['energy_bin_width']:.2e} eV.\n")
            print(f"    Parameters for wavelets:")
            if params_copy["log_wavelet_q"]:
                print(f"    Log wavelet basis in q with eps_q={params_copy['eps_q']:.2e}.")
            else:
                print(f"    Standard (linear) wavelet basis in q.")
            print(f"    Reference velocity v_max = {params_copy['v_max']:.2e}.")
            print(f"    Reference momentum q_max = {params_copy['q_max']:.2e} eV.")
            print(f"    Projecting onto basis with l_max={params_copy['l_max']}, nv_max={max(params_copy['nv_list'])}, nq_max={max(params_copy['nq_list'])}.")

            print("\n    Calculating McalI matrix coefficients...")

        for idx_bin in range(n_bins):
            if verbose:
                if idx_bin % (n_bins // 5 + 1) == 0:
                    print(f"        Projecting energy bin {idx_bin}/{n_bins-1}...")

            energy = (
                params_copy["energy_threshold"]
                + (idx_bin + 0.5) * params_copy["energy_bin_width"]
            )
            params_copy["energy"] = energy

            mcalI = McalI({}, {})
            mcalI.project(params_copy, verbose=False)
            self.mcalIs[idx_bin] = mcalI

        self.n_bins = n_bins
        self.l_max = params_copy["l_max"]
        self.nv_list = params_copy["nv_list"]
        self.nq_list = params_copy["nq_list"]
        self.v_max = params_copy["v_max"]
        self.q_max = params_copy["q_max"]
        self.fdm = params_copy["fdm"]
        self.log_wavelet_q = params_copy["log_wavelet_q"]
        self.eps_q = params_copy["eps_q"]
        if verbose:
            end_time = time.time()
            print(f"    Total projection time: {end_time - start_time:.2f} seconds.")
