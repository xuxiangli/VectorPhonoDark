import numpy as np
import numba
import time
import vsdm

from . import basis_funcs
from . import utility
from . import physics


@numba.njit
def generate_mesh_ylm_jacob(u_max, n_r, n_theta, n_phi,
                           power_r, power_theta, power_phi,
                           lm_list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate mesh points, spherical harmonic values, and Jacobian values on the mesh.

    Args:
        n_r: 
            int: Number of radial grid points.
        n_theta: 
            int: Number of theta grid points.
        n_phi: 
            int: Number of phi grid points.
        power_r: 
            float: Power for radial grid spacing.
        power_theta: 
            float: Power for theta grid spacing.
        power_phi: 
            float: Power for phi grid spacing.
        l_max: 
            int: Maximum angular quantum number.
        u_max: 
            float: Maximum radial value.
    Returns:
        tuple[np.ndarray, dict[tuple[int, int], np.ndarray], np.ndarray]: A tuple containing:
            - u_xyz_list: An array of shape (n_r*n_theta*n_phi, 3) representing Cartesian coordinates of the mesh points.
            - y_lm_vals: A dictionary with keys as (l, m) tuples and values as arrays of shape (n_theta, n_phi) representing spherical harmonic values.
            - jacob_list: An array of shape (n_r,) representing Jacobian values for integration.
    """

    dcostheta   = 2.0 / n_theta
    theta_list  = np.arccos(-np.linspace(-1. + dcostheta/2, 1. - dcostheta/2, n_theta))
    dphi        = 2 * np.pi / n_phi
    phi_list    = np.linspace(dphi/2, 2*np.pi - dphi/2, n_phi)

    y_lm_vals   = {}
    for (l, m) in lm_list:
        y_lm_vals[(l, m)] = np.array([vsdm.ylm_real(l, m, theta, phi) 
                                      for theta in theta_list for phi in phi_list]
                                     ).reshape(n_theta, n_phi)

    dx          = 1.0 / n_r
    x_list      = np.linspace(dx/2, 1.0 - dx/2, n_r)
    if power_r != 1:
        x_list      = np.power(x_list, power_r)

    if power_r == 1:
        jacob_vals  = np.array([x**2 * dx * dcostheta * dphi for x in x_list])
    else:
        jacob_vals  = np.array([x**2 * (power_r*np.power(x, 1-1/power_r)*dx) 
                                * dcostheta * dphi for x in x_list])

    u_sph_list  = np.array([[u_max * x, theta, phi] 
                            for x in x_list for theta in theta_list for phi in phi_list]
                           ).reshape(n_r*n_theta*n_phi, 3)
    u_xyz_list  = utility.sph_to_cart(u_sph_list)

    return u_xyz_list, y_lm_vals, jacob_vals


@numba.njit(fastmath=True)
def proj_integrate_3d(func_vals, haar_vals, y_lm_vals, jacob_vals) -> float:
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


@numba.njit(fastmath=True)
def proj_get_f_nlm(n_max, lm_list,
                   func_vals, y_lm_vals, jacob_vals,
                   basis, n_r, power_r, verbose=False) -> dict[tuple[int, int, int], float]:
    """
    Project function values onto basis functions to obtain f_nlm coefficients.

    Args:
        n_max: 
            int: Maximum radial quantum number.
        lm_list: 
            list[tuple[int, int]]: List of (l, m) tuples representing angular quantum numbers.
        func_vals: 
            np.ndarray: The function values on the grid.
        y_lm_vals: 
            dict[tuple[int, int], np.ndarray]: The spherical harmonic values on the grid.
        jacob_vals: 
            np.ndarray: The Jacobian values on the grid.
        basis: 
            str: The type of basis functions to use.
        n_r: 
            int: Number of radial points.
        power_r: 
            float: Power parameter for radial scaling.
        verbose: 
            bool: Whether to print verbose output.

    Returns:
        dict[tuple[int, int, int], float]: A dictionary with keys as (n, l, m) tuples and values as the corresponding f_nlm coefficients.
    """

    f_nlm = {}

    for n in range(n_max+1):

        if basis == 'haar':
            support = basis_funcs.haar_support(n)
            value = basis_funcs.haar_value(n, dim=3)
        else:
            raise NotImplementedError("Projection: Unsupported basis type.")

        r_min_idx = int(np.power(support[0], 1./power_r) * n_r)
        r_max_idx = int(np.power(support[-1], 1./power_r) * n_r)
        if n == 0:
            for l, m in lm_list:
                f_nlm[(n, l, m)] = proj_integrate_3d(
                    func_vals[r_min_idx:r_max_idx, :, :],
                    value[0],
                    y_lm_vals[(l, m)],
                    jacob_vals[r_min_idx:r_max_idx]
                )
        else:
            r_mid_idx = int(np.power(support[1], 1./power_r) * n_r)
            for l, m in lm_list:
                f_nlm[(n, l, m)] = proj_integrate_3d(
                    func_vals[r_min_idx:r_mid_idx, :, :],
                    value[0],
                    y_lm_vals[(l, m)],
                    jacob_vals[r_min_idx:r_mid_idx]
                )
                f_nlm[(n, l, m)] += proj_integrate_3d(
                    func_vals[r_mid_idx:r_max_idx, :, :],
                    value[1],
                    y_lm_vals[(l, m)],
                    jacob_vals[r_mid_idx:r_max_idx]
                )

    return f_nlm


def proj_form_factor(n_max, l_max,
                     physics_params, numerics_params, phonopy_params, file_params, 
                     phonon_file, c_dict,
                     verbose=False) -> None:
    
    if verbose:
        print("\n    Starting projection of form factor onto basis functions...")
        start_total_time = time.time()

    # Extract parameters
    basis               = numerics_params['basis']
    n_r                 = numerics_params['n_r']
    n_theta             = numerics_params['n_theta']
    n_phi               = numerics_params['n_phi']
    power_r             = numerics_params['power_r']
    power_theta         = numerics_params['power_theta']
    power_phi           = numerics_params['power_phi']

    assert basis in ['haar'], "Projection: Unsupported basis type."
    assert power_theta == 1, "Projection: Only power_theta=1 is supported for spherical Haar wavelets."
    assert power_phi == 1, "Projection: Only power_phi=1 is supported for spherical Haar wavelets."
    
    energy_threshold    = physics_params['threshold']
    energy_bin_width    = numerics_params['energy_bin_width']
    energy_max_factor   = numerics_params['energy_max_factor']
    energy_max          = physics.get_energy_max(phonon_file, factor=energy_max_factor)
    energy_bin_num      = int((energy_max - energy_threshold)/energy_bin_width) + 1

    q_max               = physics_params['q_max']
    q_cut_option        = numerics_params['q_cut']
    if q_cut_option:
        q_cut = physics.compute_q_cut(phonon_file, phonopy_params['atom_masses'])
        if q_cut < q_max:
            q_max = q_cut
            if verbose:
                print(f"    Adjusted q_max to {q_max:.4f} eV due to Debye Waller factor.")
        else:
            if verbose:
                print(f"    Using specified q_max = {q_max:.4f} eV.")

    modelname           = file_params['modelname']
    csvname             = file_params['csvname']
    write_info          = {
                            'basis': basis, 
                            'threshold': energy_threshold, 
                            'energy_bin_width': energy_bin_width,
                            'q_max': q_max, 
                            'n_r': n_r, 
                            'n_theta': n_theta, 
                            'n_phi': n_phi,
                            'power_r': power_r, 
                            'power_theta': power_theta, 
                            'power_phi': power_phi,
                        }

    if verbose:
        print(f"    Using {basis} basis with n_max={n_max}, l_max={l_max}")
        print(f"    Grid size: n_r={n_r}, n_theta={n_theta}, n_phi={n_phi}")
        print(f"    Grid rescaled: power_r={power_r}, power_theta={power_theta}, power_phi={power_phi}")
        print(f"    Energy bins: {energy_bin_num} bins from {energy_threshold} eV to {energy_max} eV")
        print("\n    Preparing grid points and basis function values...")

    # Prepare grid points and basis function values
    lm_list     = [(l, m) for l in range(l_max+1) for m in range(-l, l+1)]
    q_xyz_list, y_lm_vals, jacob_vals = generate_mesh_ylm_jacob(
        q_max, n_r, n_theta, n_phi, power_r, power_theta, power_phi, lm_list
    )

    if verbose:
        print("    Calculating form factor on grid...")
        start_time = time.time()

    # Calculate form factor on grid
    form_factor_bin_vals = physics.form_factor(q_xyz_list, 
                                               energy_threshold, energy_bin_width, energy_max,
                                               numerics_params, phonopy_params, c_dict, phonon_file
                                               ).reshape(energy_bin_num, n_r, n_theta, n_phi)
    del q_xyz_list

    if verbose:
        end_time = time.time()
        print(f"    Form factor calculation completed in {end_time - start_time:.2f} seconds.")
        print("\n    Projecting form factor onto basis functions and saving results...")

    # Project form factor onto basis functions
    for i_bin in range(energy_bin_num):

        if verbose:
            if i_bin % (energy_bin_num // 5 + 1) == 0:
                print(f"      Projecting energy bin {i_bin}/{energy_bin_num-1}...")

        f_nlm = proj_get_f_nlm(
            n_max, lm_list,
            form_factor_bin_vals[i_bin, :, :, :], y_lm_vals, jacob_vals,
            basis, n_r, power_r, verbose=verbose
        )
            
        # Save to CSV
        utility.writeFnlm_csv(csvsave_name=csvname+'_bin_'+str(i_bin)+'.csv', 
                           f_nlm_coeffs=f_nlm, 
                           info=write_info,
                           use_gvar=False)
    
    if verbose:
        print("    Projection completed for all energy bins.")
        print(f"\n    Coefficients saved to {csvname}_bin_*.csv")
        end_total_time = time.time()
        print(f"    Total projection time: {end_total_time - start_total_time:.2f} seconds.")


def proj_vdf(n_max, l_max, vdf,
             physics_params, numerics_params, file_params,
             verbose=False) -> None:
    """
    Project the velocity distribution function onto basis functions.

    Args:
        n_max: 
            int: Maximum radial quantum number.
        l_max: 
            int: Maximum angular quantum number.
        vdf: 
            function: The velocity distribution function to be projected.
        physics_params: 
            dict: Dictionary of physics parameters.
        numerics_params: 
            dict: Dictionary of numerical parameters.
        file_params: 
            dict: Dictionary of file parameters.
        verbose: 
            bool: Whether to print verbose output.

    Returns:
        None
    """

    if verbose:
        print("\n    Starting projection of velocity distribution function onto basis functions...")
        start_total_time = time.time()
    
    basis               = numerics_params['basis']
    n_r                 = numerics_params['n_r']
    n_theta             = numerics_params['n_theta']
    n_phi               = numerics_params['n_phi']
    power_r             = numerics_params['power_r']
    power_theta         = numerics_params['power_theta']
    power_phi           = numerics_params['power_phi']

    assert basis in ['haar'], "Projection: Unsupported basis type."
    assert power_theta == 1, "Projection: Only power_theta=1 is supported for spherical Haar wavelets."
    assert power_phi == 1, "Projection: Only power_phi=1 is supported for spherical Haar wavelets."

    v_max               = physics_params['v_max']
    vdf_params          = physics_params['vdf_params']

    modelname           = file_params['vdf_model']
    csvname             = file_params['csvname']
    write_info          = {
                            'basis': basis, 
                            'v_max': v_max, 
                            'n_r': n_r, 
                            'n_theta': n_theta, 
                            'n_phi': n_phi,
                            'power_r': power_r, 
                            'power_theta': power_theta, 
                            'power_phi': power_phi,
                            'vdf_model': modelname,
                            'vdf_params': vdf_params,
                        }

    if verbose:
        print(f"    Using {basis} basis with n_max={n_max}, l_max={l_max}")
        print(f"    Grid size: n_r={n_r}, n_theta={n_theta}, n_phi={n_phi}")
        print(f"    Grid rescaled: power_r={power_r}, power_theta={power_theta}, power_phi={power_phi}")
        print("\n    Preparing grid points and basis function values...")

    # Prepare grid points and basis function values
    lm_list     = [(l, m) for l in range(l_max+1) for m in range(-l, l+1)]
    v_xyz_list, y_lm_vals, jacob_vals = generate_mesh_ylm_jacob(
        v_max, n_r, n_theta, n_phi, power_r, power_theta, power_phi, lm_list
    )

    if verbose:
        print("    Calculating form factor on grid...")
        start_time = time.time()
    
    # Calculate vdf on grid
    vdf_vals = np.array([vdf(v_vec, *vdf_params) for v_vec in v_xyz_list]
                        ).reshape(n_r, n_theta, n_phi)
    del v_xyz_list

    if verbose:
        end_time = time.time()
        print(f"    VDF calculation completed in {end_time - start_time:.2f} seconds.")
        print("\n    Projecting VDF onto basis functions and saving results...")

    # Project vdf onto basis functions
    f_nlm = proj_get_f_nlm(
        n_max, lm_list,
        vdf_vals, y_lm_vals, jacob_vals,
        basis, n_r, power_r, verbose=verbose
    )

    # Save to CSV
    utility.writeFnlm_csv(csvsave_name=csvname+'.csv', 
                       f_nlm_coeffs=f_nlm, 
                       info=write_info,
                       use_gvar=False)
    
    if verbose:
        print("    Projection completed.")
        print(f"    Coefficients saved to {csvname}.csv")
        end_total_time = time.time()
        print(f"    Total projection time: {end_total_time - start_total_time:.2f} seconds.")