import numpy as np
import numba
import sys
import quaternionic
from importlib import util

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
    return 1 / quaternionic.array(qr, qi, qj, qk)


def euler_to_quaternion(alpha, beta, gamma=0.0):
    """
    Convert Euler angles (alpha, beta, gamma) to quaternion representation.
    """

    w = np.cos(beta / 2.0) * np.cos((alpha + gamma) / 2.0)
    x = -np.sin(beta / 2.0) * np.sin((alpha - gamma) / 2.0)
    y = np.sin(beta / 2.0) * np.cos((alpha - gamma) / 2.0)
    z = np.cos(beta / 2.0) * np.sin((alpha + gamma) / 2.0)

    return quaternionic.array(w, x, y, z)


def rot_to_quaternion(nx, ny, nz, alpha):
    """
    Get quaternionic representation of rotation given by axis (nx, ny, nz) and angle theta.
    """

    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    if norm == 0:
        raise ValueError("Rotation axis cannot be the zero vector.")
    nx /= norm
    ny /= norm
    nz /= norm

    half_alpha = alpha / 2
    w = np.cos(half_alpha)
    x = nx * np.sin(half_alpha)
    y = ny * np.sin(half_alpha)
    z = nz * np.sin(half_alpha)
    return quaternionic.array(w, x, y, z)


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
    u_xyz_list = sph_to_cart(u_sph_list)

    return u_xyz_list, y_lm_vals, jacob_vals


@numba.njit
def gen_log_mesh_ylm_jacob(
    lm_list: list[tuple[int, int]],
    u_max: float,
    n_a: int,
    n_b: int,
    n_c: int,
    eps: float = 0.0,
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

    length = -np.log(eps)
    r_list = np.exp(length * a_list + np.log(eps))
    dr_da = length * r_list
    jacob_list = r_list**2 * (dr_da * da) * dcostheta * dphi

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

    if n == 0:
        r_min_idx = 0
        r_mid_idx = n_a
        r_max_idx = n_a
    else:
        x_min, x_mid, x_max = basis_funcs.haar_support_log(n, eps)
        length = np.log(1.0 / eps)
        # r_min_idx = int((np.log(x_min / eps) / (length / n_a)))
        # r_mid_idx = int((np.log(x_mid / eps) / (length / n_a)))
        # r_max_idx = int((np.log(x_max / eps) / (length / n_a)))
        r_min_idx = int(round(np.log(x_min / eps) / (length / n_a)))
        r_mid_idx = int(round(np.log(x_mid / eps) / (length / n_a)))
        r_max_idx = int(round(np.log(x_max / eps) / (length / n_a)))

        r_min_idx = max(0, min(r_min_idx, n_a))
        r_mid_idx = max(r_min_idx, min(r_mid_idx, n_a))
        r_max_idx = max(r_mid_idx, min(r_max_idx, n_a))

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
    total = 0.0
    for i in range(n_r):
        temp = 0.0
        for j in range(n_theta):
            for k in range(n_phi):
                temp += func_vals[i, j, k] * y_lm_vals[j, k]
        total += temp * jacob_vals[i]

    # weighted_vals = func_vals * y_lm_vals[np.newaxis, :, :]
    # # angular_integral = np.sum(weighted_vals, axis=(1, 2))
    # temp_sum = np.sum(weighted_vals, axis=2)
    # angular_integral = np.sum(temp_sum, axis=1)
    # total = np.sum(angular_integral * jacob_vals)

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
    eps: float = 1.0,
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
            r_min_idx, r_mid_idx, r_max_idx = get_wavelet_boundary_log(n, n_a, eps=eps)
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


def gen_quad_angular_grid(
    lm_list: list[tuple[int, int]],
    n_gl: int,
    n_phi: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Precompute angular quadrature nodes and weighted spherical harmonics
    for the quad-based projection.

    Uses Gauss-Legendre quadrature for cos(theta) and uniform nodes for phi.
    Analogous to gen_mesh_ylm_jacob but for the quad-based method.

    Parameters
    ----------
    lm_list : list of tuples
        List of (l, m) tuples.
    n_gl : int
        Number of Gauss-Legendre nodes for cos(theta).
    n_phi : int
        Number of uniform nodes for phi.

    Returns
    -------
    hat_v : np.ndarray
        Unit direction vectors at angular nodes, shape (n_gl * n_phi, 3).
    ylm_weighted : np.ndarray
        Spherical harmonics weighted by GL weights, shape (n_lm, n_gl * n_phi).
    dphi : float
        Uniform phi step size (phi quadrature weight).
    """
    cos_theta_nodes, cos_theta_weights = np.polynomial.legendre.leggauss(n_gl)
    theta_nodes = np.arccos(cos_theta_nodes)
    sin_theta_nodes = np.sqrt(1.0 - cos_theta_nodes**2)

    dphi = 2 * np.pi / n_phi
    phi_nodes = np.linspace(dphi / 2, 2 * np.pi - dphi / 2, n_phi)
    cos_phi_nodes = np.cos(phi_nodes)
    sin_phi_nodes = np.sin(phi_nodes)

    # Unit direction vectors: hat_v[i] = (sin_theta*cos_phi, sin_theta*sin_phi, cos_theta)
    hat_v = np.empty((n_gl * n_phi, 3), dtype=np.float64)
    for i_th in range(n_gl):
        for i_phi in range(n_phi):
            idx = i_th * n_phi + i_phi
            hat_v[idx, 0] = sin_theta_nodes[i_th] * cos_phi_nodes[i_phi]
            hat_v[idx, 1] = sin_theta_nodes[i_th] * sin_phi_nodes[i_phi]
            hat_v[idx, 2] = cos_theta_nodes[i_th]

    ang_weights = np.repeat(cos_theta_weights, n_phi)

    ylm_flat = np.zeros((len(lm_list), n_gl * n_phi), dtype=np.float64)
    for idx_lm, (l, m) in enumerate(lm_list):
        for i_th in range(n_gl):
            for i_phi in range(n_phi):
                idx = i_th * n_phi + i_phi
                ylm_flat[idx_lm, idx] = vsdm.ylm_real(
                    l, m, theta_nodes[i_th], phi_nodes[i_phi]
                )

    ylm_weighted = ylm_flat * ang_weights[np.newaxis, :]
    return hat_v, ylm_weighted, dphi


def proj_get_f_lm_n_quad(
    n_max: int,
    lm_list: list[tuple[int, int]],
    vdf_func,
    vdf_params: dict,
    v_max: float,
    hat_v: np.ndarray,
    ylm_weighted: np.ndarray,
    dphi: float,
    epsabs: float = 1e-8,
    epsrel: float = 1e-8,
    limit: int = 200,
    verbose: bool = False,
) -> np.ndarray:
    """
    Project a VDF onto the spherical wavelet basis using adaptive radial
    quadrature (scipy.integrate.quad).

    Analogous to proj_get_f_lm_n but uses scipy.integrate.quad for the
    radial integral instead of a precomputed grid sum.

    Parameters
    ----------
    n_max : int
        Maximum radial Haar wavelet order.
    lm_list : list of tuples
        List of (l, m) tuples.
    vdf_func : callable
        The VDF function, must accept (v_xyz, **vdf_params).
    vdf_params : dict
        Parameters passed to vdf_func.
    v_max : float
        Velocity scale; physical velocity is v_max * r * hat_v.
    hat_v : np.ndarray
        Unit direction vectors at angular nodes, shape (n_ang, 3).
    ylm_weighted : np.ndarray
        GL-weighted spherical harmonics, shape (n_lm, n_ang).
    dphi : float
        Uniform phi step size (phi quadrature weight).
    epsabs : float
        Absolute tolerance for scipy.integrate.quad.
    epsrel : float
        Relative tolerance for scipy.integrate.quad.
    limit : int
        Max adaptive subdivisions for scipy.integrate.quad.
    verbose : bool
        If True, print progress per wavelet index n.

    Returns
    -------
    np.ndarray
        Array of shape (n_lm, n_max + 1) with the projection coefficients.
    """
    import warnings
    from scipy import integrate

    n_ang = hat_v.shape[0]
    f_lm_n = np.zeros((len(lm_list), n_max + 1), dtype=np.float64)

    for n in range(n_max + 1):

        if n == 0:
            intervals = [(0.0, 1.0)]
            haar_vals = [basis_funcs.haar_value(0, dim=3)[0]]
        else:
            x_min, x_mid, x_max = basis_funcs.haar_support(n)
            a_n, neg_b_n = basis_funcs.haar_value(n, dim=3)
            intervals = [(x_min, x_mid), (x_mid, x_max)]
            haar_vals = [a_n, neg_b_n]

        for idx_lm in range(len(lm_list)):
            total = 0.0
            ylm_w = ylm_weighted[idx_lm]

            for (r_lo, r_hi), h_val in zip(intervals, haar_vals):
                if r_hi <= r_lo:
                    continue

                def radial_integrand(r, _ylm_w=ylm_w, _h=h_val):
                    v_xyz_all = v_max * r * hat_v
                    vdf_vals = np.array([
                        vdf_func(v_xyz_all[i], **vdf_params) for i in range(n_ang)
                    ])
                    angular_sum = np.dot(vdf_vals, _ylm_w) * dphi
                    return angular_sum * r**2 * _h

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", integrate.IntegrationWarning)
                    val, _ = integrate.quad(
                        radial_integrand, r_lo, r_hi,
                        epsabs=epsabs, epsrel=epsrel, limit=limit,
                    )
                total += val

            f_lm_n[idx_lm, n] = total

    return f_lm_n
