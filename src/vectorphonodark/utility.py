import numpy as np
import numba
import sys
import quaternionic
from importlib import util
from functools import reduce

import vsdm

from . import basis_funcs


# Define color codes for terminal output
if sys.stdout.isatty():
    C_GREEN = "\033[92m"
    C_CYAN = "\033[96m"
    C_RESET = "\033[0m"
else:
    C_GREEN = ""
    C_CYAN = ""
    C_RESET = ""


def import_file(full_name, path):
    """
    Import a module from a given file path.

    Parameters
    ----------
    full_name : str
        The full name to assign to the module.
    path : str
        The file path to the module.

    Returns
    -------
    module
        The imported module.
    """

    spec = util.spec_from_file_location(full_name, path)
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


@numba.njit
def sph_to_cart(vec_sph) -> np.ndarray:
    """
    Convert spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    vec_sph : np.ndarray
        An array of shape (..., 3) representing points in spherical coordinates 
        (r, theta, phi).

    Returns
    -------
    np.ndarray
        An array of shape (..., 3) representing points in Cartesian coordinates 
        (x, y, z).
    """
    r = vec_sph[..., 0]
    theta = vec_sph[..., 1]
    phi = vec_sph[..., 2]

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.stack((x, y, z), axis=-1)


# def get_intersection_index(base=None, **lists):
#     """
#     Get the indices of common elements in multiple lists.

#     If `base` is provided, it is used as the reference list to find common elements.
#     Otherwise, the first list in `lists` is used as the reference.

#     Parameters
#     ----------
#     base : list, optional
#         The base list to compare against.
#     **lists : dict
#         Arbitrary number of lists to find common elements with.

#     Returns
#     -------
#     np.ndarray
#         Array of indices in each list corresponding to the common elements.
#             shape: (number of lists, number of common elements)
#     """
#     arrays = [np.asarray(l) for l in lists.values()]
#     base_arr = np.asarray(base) if base is not None else np.array([])
#     if base_arr.size > 0:
#         arrays.insert(0, base_arr)

#     if not arrays:
#         return np.array([])

#     common_elements = reduce(np.intersect1d, arrays)

#     if common_elements.size == 0:
#         return np.empty((len(lists), 0), dtype=int)

#     target_arrays = [np.asarray(l) for l in lists.values()]
#     indices = np.vstack(
#         [np.searchsorted(arr, common_elements) for arr in target_arrays]
#     )
#     return indices


def getQ(theta, phi):
    """
    Get quaternionic representation of rotation given by Euler angles (theta, phi).
    """

    axisphi = phi + np.pi / 2  # stationary under R
    axR = theta / 2
    qr = np.cos(axR)
    qi = np.sin(axR) * np.cos(axisphi)
    qj = np.sin(axR) * np.sin(axisphi)
    qk = 0.0
    return quaternionic.array(qr, qi, qj, qk)


@numba.njit
def gen_mesh_ylm_jacob(
    lm_list: list[tuple[int, int]],
    u_max: float,
    n_a: int,
    n_b: int,
    n_c: int,
    power_a: float = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate power-spaced mesh points, spherical harmonic values, and Jacobian.

    The mesh points are taken in between 0 and u_max in the radial direction,
    and are evenly spaced in theta and phi directions. 
    
    The radial grid can be linear or power-law spaced depending on the power_a 
    parameter.

    Parameters
    ----------
    lm_list : list of tuples
        List of (l, m) tuples.
    u_max : float
        Maximum radial value.
    n_a : int
        Number of radial grid points.
    n_b : int
        Number of theta grid points.
    n_c : int
        Number of phi grid points.
    power_a : float
        Power for radial grid spacing.

    Returns
    -------
    u_xyz_list : np.ndarray
        An array of shape (n_a*n_b*n_c, 3) representing Cartesian coordinates 
        of the mesh points.
    y_lm_vals : dict
        A dictionary with keys as (l, m) tuples and values as arrays of shape 
        (n_b, n_c) representing spherical harmonic values.
    jacob_list : np.ndarray
        An array of shape (n_a,) representing Jacobian values for integration.
    """

    dcostheta = 2.0 / n_b
    theta_list = np.arccos(
        -np.linspace(-1.0 + dcostheta / 2, 1.0 - dcostheta / 2, n_b)
    )
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
    u_xyz_list = sph_to_cart(u_sph_list)

    return u_xyz_list, y_lm_vals, jacob_vals


@numba.njit
def gen_log_mesh_ylm_jacob(
    lm_list: list[tuple[int, int]],
    u_max: float,
    n_a: int,
    n_b: int,
    n_c: int,
    eps: float = 0.,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate log-spaced mesh points, spherical harmonic values, and Jacobian.

    The mesh points are taken evenly in logarithmic scale in between eps*u_max 
    and u_max in the radial direction, and are evenly spaced in theta and phi 
    directions.

    Parameters
    ----------
    lm_list : list of tuples
        List of (l, m) tuples.
    u_max : float
        Maximum radial value.
    n_a : int
        Number of radial grid points.
    n_b : int
        Number of theta grid points.
    n_c : int
        Number of phi grid points.
    eps : float
        Minimum radial value, taken in (0, 1).

    Returns
    -------
    u_xyz_list : np.ndarray
        An array of shape (n_a*n_b*n_c, 3) representing Cartesian coordinates 
        of the mesh points.
    y_lm_vals : dict
        A dictionary with keys as (l, m) tuples and values as arrays of shape 
        (n_b, n_c) representing spherical harmonic values.
    jacob_list : np.ndarray
        An array of shape (n_a,) representing Jacobian values for integration.
    """

    dcostheta = 2.0 / n_b
    theta_list = np.arccos(
        -np.linspace(-1.0 + dcostheta / 2, 1.0 - dcostheta / 2, n_b)
    )
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
    u_xyz_list = sph_to_cart(u_sph_list)

    return u_xyz_list, y_lm_vals, jacob_list


@numba.njit
def get_wavelet_boundary(n, n_a, power_a=1) -> tuple[int, int, int]:
    """
    Get the boundary of the order n wavelet in radial direction of the mesh
    with n_a points and power scaling power_a.

    Parameters
    ----------
    n : int
        Wavelet order.
    n_a : int
        Number of radial points.
    power_a : float, optional
        Power for radial grid spacing. Default is 1.

    Returns
    -------
    tuple[int, int, int]
        The (r_min_idx, r_mid_idx, r_max_idx) indices for the wavelet boundaries.

    Examples
    --------
    >>> n_a = 16
    >>> power_a = 1
    >>> for n in range(5):
    ...     r_min_idx, r_mid_idx, r_max_idx = get_wavelet_boundary(n, n_a, power_a)
    ...     print(f"n={n}: r_min_idx={r_min_idx}, r_mid_idx={r_mid_idx}, r_max_idx={r_max_idx}")
    n=0: r_min_idx=0, r_mid_idx=16, r_max_idx=16
    n=1: r_min_idx=0, r_mid_idx=8, r_max_idx=16
    n=2: r_min_idx=8, r_mid_idx=12, r_max_idx=16
    n=3: r_min_idx=4, r_mid_idx=6, r_max_idx=8
    n=4: r_min_idx=6, r_mid_idx=7, r_max_idx=8
    """

    if n == 0:
        r_min_idx = 0
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
def get_wavelet_boundary_log(n, n_a, eps) -> tuple[int, int, int]:
    """
    Get the boundary of the order n wavelet in radial direction of the log-spaced
    mesh with n_a points between eps and 1.0.

    Parameters
    ----------
    n : int
        Wavelet order.
    n_a : int
        Number of radial points.
    eps : float
        Minimum radial value, taken in (0, 1).

    Returns
    -------
    tuple[int, int, int]
        The (r_min_idx, r_mid_idx, r_max_idx) indices for the wavelet boundaries.
    """

    if n == 0:
        r_min_idx = 0
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

    Shape of func_vals: (n_r, n_theta, n_phi)
    Shape of y_lm_vals: (n_theta, n_phi)
    Shape of jacob_vals: (n_r)

    Parameters
    ----------
    func_vals : np.ndarray
        The function values on the 3d grid.
    haar_vals : float
        The Haar basis function values in the region.
    y_lm_vals : np.ndarray
        The spherical harmonic values on the angular grid.
    jacob_vals : np.ndarray
        The Jacobian values on the radial grid.

    Returns
    -------
    float
        The result of the 3D integration.
    """
    n_r, n_theta, n_phi = func_vals.shape
    assert y_lm_vals.shape == (n_theta, n_phi)
    assert jacob_vals.shape[0] == n_r
    total = 0.0
    for i in range(n_r):
        temp = 0.0
        for j in range(n_theta):
            for k in range(n_phi):
                temp += func_vals[i, j, k] * y_lm_vals[j, k]
        total += temp * jacob_vals[i]
    return total * haar_vals


@numba.njit
def proj_get_f_lm_n(
    n_max: int,
    lm_list: list[tuple[int, int]],
    func_vals: np.ndarray,
    y_lm_vals: dict[tuple[int, int], np.ndarray],
    jacob_vals: np.ndarray,
    n_a: int,
    power_a: float = 1,
    log_wavelet: bool = False,
    eps: float = 1.,
) -> dict[tuple[int, int, int], float]:
    """
    Project function values onto basis functions to obtain f_nlm coefficients.

    Parameters
    ----------
    n_max : int
        Maximum radial Haar wavelet order.
    lm_list : list[tuple[int, int]]
        List of (l, m) tuples representing angular quantum numbers.
    func_vals : np.ndarray
        The function values on the 3d grid.
    y_lm_vals : dict[tuple[int, int], np.ndarray]
        The spherical harmonic values on the angular grid.
    jacob_vals : np.ndarray
        The Jacobian values on the radial grid.
    n_a : int
        Number of radial points.
    power_a : float
        Power parameter for radial scaling.
    log_wavelet : bool
        Whether to use log-spaced wavelets.
        If True, power_a is ignored and eps is used.
    eps : float
        Minimum radial value for log-spaced wavelets.

    Returns
    -------
    dict[tuple[int, int, int], float]
        A dictionary with keys as (n, l, m) tuples and values as the 
        corresponding f_nlm coefficients.
    """

    f_lm_n = np.zeros((len(lm_list), n_max + 1), dtype=np.float64)

    for n in numba.prange(n_max + 1):

        # Get wavelet values and boundaries
        if log_wavelet:
            value = basis_funcs.haar_value_log(n, eps=eps, p=2)
            r_min_idx, r_mid_idx, r_max_idx = get_wavelet_boundary_log(
                n, n_a, eps=eps
            )   
        else:
            value = basis_funcs.haar_value(n, dim=3)
            r_min_idx, r_mid_idx, r_max_idx = get_wavelet_boundary(
                n, n_a, power_a=power_a
            )

        # Perform integration
        if n == 0:
            for idx_lm, (l, m) in enumerate(lm_list):
                f_lm_n[idx_lm, n] = proj_integrate_3d(
                    func_vals[r_min_idx:r_max_idx, :, :],
                    value[0],
                    y_lm_vals[(l, m)],
                    jacob_vals[r_min_idx:r_max_idx],
                )
        else:
            for idx_lm, (l, m) in enumerate(lm_list):
                f_lm_n[idx_lm, n] = proj_integrate_3d(
                    func_vals[r_min_idx:r_mid_idx, :, :],
                    value[0],
                    y_lm_vals[(l, m)],
                    jacob_vals[r_min_idx:r_mid_idx],
                )
                f_lm_n[idx_lm, n] += proj_integrate_3d(
                    func_vals[r_mid_idx:r_max_idx, :, :],
                    value[1],
                    y_lm_vals[(l, m)],
                    jacob_vals[r_mid_idx:r_max_idx],
                )

    return f_lm_n