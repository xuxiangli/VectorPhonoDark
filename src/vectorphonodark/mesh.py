"""Angular/radial meshes, wavelet boundaries, and projection integrators.

Pure numerical helpers shared by the projection classes; no package state.
"""

import numba
import numpy as np
import vsdm

from . import basis_funcs


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


@numba.njit
def gen_mesh_ylm_jacob(
    lm_list: np.ndarray,
    u_max: float,
    n_a: int,
    n_b: int,
    n_c: int,
    power_a: float = 1,
) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray], np.ndarray]:
    """
    Generate power-spaced mesh points, spherical harmonic values, and Jacobian.

    The mesh points are taken in between 0 and u_max in the radial direction,
    and are evenly spaced in theta and phi directions.

    The radial grid can be linear or power-law spaced depending on the power_a
    parameter.

    Parameters
    ----------
    lm_list : np.ndarray
        (n_lm, 2) int array of (l, m) pairs.
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
    jacob_vals : np.ndarray
        An array of shape (n_a,) representing Jacobian values for integration.
    """

    dcostheta = 2.0 / n_b
    theta_list = np.arccos(-np.linspace(-1.0 + dcostheta / 2, 1.0 - dcostheta / 2, n_b))
    dphi = 2 * np.pi / n_c
    phi_list = np.linspace(dphi / 2, 2 * np.pi - dphi / 2, n_c)

    y_lm_vals = {}
    for ell, m in lm_list:
        y_lm_vals[(ell, m)] = np.array(
            [
                vsdm.ylm_real(ell, m, theta, phi)
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
def gen_mesh_ylm_jacob_log(
    lm_list: np.ndarray,
    u_max: float,
    n_a: int,
    n_b: int,
    n_c: int,
    eps: float,
) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray], np.ndarray]:
    """
    Generate log-spaced mesh points, spherical harmonic values, and Jacobian.

    The mesh points are taken evenly in logarithmic scale in between eps*u_max
    and u_max in the radial direction, and are evenly spaced in theta and phi
    directions.

    Parameters
    ----------
    lm_list : np.ndarray
        (n_lm, 2) int array of (l, m) pairs.
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
    jacob_vals : np.ndarray
        An array of shape (n_a,) representing Jacobian values for integration.
    """

    if not (0.0 < eps < 1.0):
        raise ValueError("gen_mesh_ylm_jacob_log: eps must be in (0, 1).")

    dcostheta = 2.0 / n_b
    theta_list = np.arccos(-np.linspace(-1.0 + dcostheta / 2, 1.0 - dcostheta / 2, n_b))
    dphi = 2 * np.pi / n_c
    phi_list = np.linspace(dphi / 2, 2 * np.pi - dphi / 2, n_c)

    y_lm_vals = {}
    for ell, m in lm_list:
        y_lm_vals[(ell, m)] = np.array(
            [
                vsdm.ylm_real(ell, m, theta, phi)
                for theta in theta_list
                for phi in phi_list
            ]
        ).reshape(n_b, n_c)

    da = 1.0 / n_a
    a_list = np.linspace(da / 2, 1.0 - da / 2, n_a)

    length = -np.log(eps)
    r_list = np.exp(length * a_list + np.log(eps))
    dr_da = length * r_list
    jacob_vals = r_list**2 * (dr_da * da) * dcostheta * dphi

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
def get_wavelet_boundary(n, n_a, power_a: float = 1) -> tuple[int, int, int]:
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
    ...     print(
    ...         f"n={n}: r_min_idx={r_min_idx}, "
    ...         f"r_mid_idx={r_mid_idx}, r_max_idx={r_max_idx}"
    ...     )
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
            r_min_idx = int(round(x_min * n_a))
            r_mid_idx = int(round(x_mid * n_a))
            r_max_idx = int(round(x_max * n_a))
        else:
            r_min_idx = int(round(np.power(x_min, 1.0 / power_a) * n_a))
            r_mid_idx = int(round(np.power(x_mid, 1.0 / power_a) * n_a))
            r_max_idx = int(round(np.power(x_max, 1.0 / power_a) * n_a))

        r_min_idx = max(0, min(r_min_idx, n_a))
        r_mid_idx = max(r_min_idx, min(r_mid_idx, n_a))
        r_max_idx = max(r_mid_idx, min(r_max_idx, n_a))

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

    if not (0.0 < eps < 1.0):
        raise ValueError("get_wavelet_boundary_log: eps must be in (0, 1).")

    if n == 0:
        r_min_idx = 0
        r_mid_idx = n_a
        r_max_idx = n_a
    else:
        x_min, x_mid, x_max = basis_funcs.haar_support_log(n, eps)
        length = np.log(1.0 / eps)
        r_min_idx = int(round(np.log(x_min / eps) / (length / n_a)))
        r_mid_idx = int(round(np.log(x_mid / eps) / (length / n_a)))
        r_max_idx = int(round(np.log(x_max / eps) / (length / n_a)))

        r_min_idx = max(0, min(r_min_idx, n_a))
        r_mid_idx = max(r_min_idx, min(r_mid_idx, n_a))
        r_max_idx = max(r_mid_idx, min(r_max_idx, n_a))

    return r_min_idx, r_mid_idx, r_max_idx


# parallel=True does not compile here
@numba.njit
def proj_get_f_lm_n(
    n_max: int,
    lm_list: np.ndarray,
    func_vals: np.ndarray,
    y_lm_vals: dict[tuple[int, int], np.ndarray],
    jacob_vals: np.ndarray,
    n_a: int,
    power_a: float = 1,
    log_wavelet: bool = False,
    eps: float = 1.0,
) -> np.ndarray:
    """
    Project function values onto basis functions to obtain f_nlm coefficients,
    with f_lm_n = sum func * haar_n * y_lm * jacob

    Parameters
    ----------
    n_max : int
        Maximum radial Haar wavelet order.
    lm_list : np.ndarray
        (n_lm, 2) int array of (l, m) pairs representing angular quantum numbers.
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
    np.ndarray
        A ``(n_lm, n_max + 1)`` array of f_nlm coefficients: row ``i`` matches
        ``lm_list[i]``, column ``n`` the radial wavelet order.
    """

    n_lm = len(lm_list)
    n_r, n_theta, n_phi = func_vals.shape

    # int_over_ang_vals[i, lm] = jacob[i] * sum_jk func_vals[i,j,k] * Y_lm[j,k]
    #     -> f_lm_n[lm, n] = haar[n] * sum_(i in sup) int_over_ang_vals[i, lm]
    #
    # func_vals.reshape (n_r, n_theta * n_phi)
    # y_lm_vals_flat    (n_lm, n_theta * n_phi)
    # jacob_vals        (n_r,)
    # int_over_ang_vals (n_r, n_lm)
    y_lm_vals_flat = np.empty((n_lm, n_theta * n_phi), dtype=np.float64)
    for idx_lm in range(n_lm):
        ell, m = lm_list[idx_lm]
        y_lm_vals_flat[idx_lm] = y_lm_vals[(int(ell), int(m))].reshape(n_theta * n_phi)

    int_over_ang_vals = func_vals.reshape(n_r, n_theta * n_phi) @ y_lm_vals_flat.T
    int_over_ang_vals *= jacob_vals[:, np.newaxis]

    f_lm_n = np.zeros((n_lm, n_max + 1), dtype=np.float64)

    for n in range(n_max + 1):
        # Get wavelet values and boundaries
        if log_wavelet:
            haar_vals = basis_funcs.haar_value_log(n, eps=eps, p=2)
            r_min_idx, r_mid_idx, r_max_idx = get_wavelet_boundary_log(n, n_a, eps=eps)
        else:
            haar_vals = basis_funcs.haar_value(n, dim=3)
            r_min_idx, r_mid_idx, r_max_idx = get_wavelet_boundary(
                n, n_a, power_a=power_a
            )

        # Sum int_over_ang_vals over each Haar region
        f_lm_n[:, n] = haar_vals[0] * int_over_ang_vals[r_min_idx:r_mid_idx].sum(
            axis=0
        ) + haar_vals[1] * int_over_ang_vals[r_mid_idx:r_max_idx].sum(axis=0)

    return f_lm_n


def _validate_radial_grid(n_a, n_max):
    """Require the radial wavelet grid to be a power of two greater than n_max."""
    is_pow2 = isinstance(n_a, (int, np.integer)) and n_a > 0 and (n_a & (n_a - 1)) == 0
    if not is_pow2 or n_a <= n_max:
        next_pow2 = 1 << int(n_max).bit_length()  # smallest power of two > n_max
        raise ValueError(
            f"radial grid n_a={n_a!r} must be a power of two greater than "
            f"n_max={n_max}; use n_a={next_pow2}."
        )
