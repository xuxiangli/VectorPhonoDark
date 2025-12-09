import numpy as np
import numba
import time
import vsdm

from . import constants as const
from . import basis_funcs
from . import utility
from . import physics


@numba.njit
def generate_mesh_ylm_jacob(lm_list: list[tuple[int, int]], u_max: float,
                            n_a: int, n_b: int, n_c: int,
                            power_a: float, power_b: float = 1, power_c: float = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        power_b: 
            float: Power for theta grid spacing.
        power_c: 
            float: Power for phi grid spacing.
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
    theta_list = np.arccos(
        -np.linspace(-1. + dcostheta / 2, 1. - dcostheta/2, n_b)
    )
    dphi = 2 * np.pi / n_c
    phi_list = np.linspace(dphi/2, 2*np.pi - dphi/2, n_c)

    y_lm_vals = {}
    for (l, m) in lm_list:
        y_lm_vals[(l, m)] = np.array([
            vsdm.ylm_real(l, m, theta, phi) for theta in theta_list for phi in phi_list]
        ).reshape(n_b, n_c)

    da = 1.0 / n_a
    a_list = np.linspace(da/2, 1.0 - da/2, n_a)
    if power_a == 1:
        dr = da
        r_list = a_list
        jacob_vals = r_list**2 * dr * dcostheta * dphi
    else:
        dr_list = power_a * np.power(a_list, power_a - 1) * da
        r_list = np.power(a_list, power_a)
        jacob_vals = r_list**2 * dr_list * dcostheta * dphi

    u_sph_list = np.array([[u_max * r, theta, phi]
                           for r in r_list for theta in theta_list for phi in phi_list]
                          ).reshape(n_a*n_b*n_c, 3)
    u_xyz_list = utility.sph_to_cart(u_sph_list)

    return u_xyz_list, y_lm_vals, jacob_vals


@numba.njit(fastmath=True)
def generate_special_mesh_jacob_nlist(u_max: float, u_min: float,
                                      n_b: int, n_c: int, lam_start: int = 0,
                                      verbose: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate special mesh points, Jacobian values, and n_list for projection.

    Args:
        u_max: 
            float: Maximum radial value.
        u_min: 
            float: Minimum radial value.
        n_a: 
            int: Number of radial grid points.
        n_b: 
            int: Number of theta grid points.
        n_c: 
            int: Number of phi grid points.
        lam_start: 
            int: Starting index for exponential scaling.
    Returns:
        u_xyz_list: An array of shape (n_a*n_b*n_c, 3) representing Cartesian coordinates of the special mesh points.
        jacob_vals: An array of shape (n_a,) representing Jacobian values for integration on the special mesh.
        n_list: An array of shape (n_a-1,) representing the n values for the wavelets.
    """

    dcostheta = 2.0 / n_b
    theta_list = np.arccos(
        -np.linspace(-1. + dcostheta / 2, 1. - dcostheta/2, n_b)
    )
    dphi = 2 * np.pi / n_c
    phi_list = np.linspace(dphi/2, 2*np.pi - dphi/2, n_c)

    lam_end = int(np.floor(np.log2(u_max / u_min)))
    if lam_end <= lam_start:
        if verbose:
            print("    Special mesh: n_max large enough, no special mesh generated.")
        return np.empty((0, 3)), np.empty((0,)), np.empty((0,), dtype=np.int64)
    n_a = lam_end - lam_start + 1

    r_list = np.power(2., -(lam_end+1) + np.arange(n_a) + 0.5)
    dr_list = r_list * np.log(2)
    jacob_vals = r_list**2 * dr_list * dcostheta * dphi

    u_sph_list = np.array([
        [u_max * r, theta, phi]
        for r in r_list for theta in theta_list for phi in phi_list
    ]).reshape(n_a*n_b*n_c, 3)
    u_xyz_list = utility.sph_to_cart(u_sph_list)

    # Only n_a-1 wavelets
    n_list = np.power(2, np.arange(lam_start, lam_end))
    if verbose:
        print(
            f"    Special mesh: Generated {len(n_list)} additional wavelets from n={n_list[0]} to n={n_list[-1]}.")

    return u_xyz_list, jacob_vals, n_list


@numba.njit(fastmath=True)
def proj_integrate_3d(func_vals: np.ndarray, haar_vals: float,
                      y_lm_vals: np.ndarray, jacob_vals: np.ndarray) -> float:
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
def proj_get_f_nlm(n_list: np.ndarray, lm_list: list[tuple[int, int]],
                   func_vals: np.ndarray,
                   y_lm_vals: dict[tuple[int, int], np.ndarray],
                   jacob_vals: np.ndarray,
                   basis: str, n_a: int, power_a: float,
                   special_mesh: bool = False, verbose: bool = False) -> dict[tuple[int, int, int], float]:
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
        n_a: 
            int: Number of radial points.
        power_a: 
            float: Power parameter for radial scaling.
        special_mesh: 
            bool: Whether a special mesh is used.
        verbose: 
            bool: Whether to print verbose output.

    Returns:
        dict[tuple[int, int, int], float]: A dictionary with keys as (n, l, m) tuples and values as the corresponding f_nlm coefficients.
    """

    f_nlm = {}

    for i_n, n in enumerate(n_list):

        if basis == 'haar':
            support = basis_funcs.haar_support(n)
            value = basis_funcs.haar_value(n, dim=3)
        else:
            raise NotImplementedError("Projection: Unsupported basis type.")

        if special_mesh:
            r_min_idx = 0
            r_mid_idx = n_a - i_n - 1
            r_max_idx = n_a - i_n
        else:
            if power_a == 1:
                r_min_idx = int(support[0] * n_a)
                r_mid_idx = int(support[1] * n_a)
                r_max_idx = int(support[-1] * n_a)
            else:
                r_min_idx = int(np.power(support[0], 1./power_a) * n_a)
                r_mid_idx = int(np.power(support[1], 1./power_a) * n_a)
                r_max_idx = int(np.power(support[-1], 1./power_a) * n_a)

        if n == 0:
            for l, m in lm_list:
                f_nlm[(n, l, m)] = proj_integrate_3d(
                    func_vals[r_min_idx:r_max_idx, :, :],
                    value[0],
                    y_lm_vals[(l, m)],
                    jacob_vals[r_min_idx:r_max_idx]
                )
        else:
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


def proj_form_factor(physics_params, numerics_params, phonopy_params, file_params,
                     phonon_file, c_dict,
                     verbose=False) -> None:

    if verbose:
        print("\n    Starting projection of form factor onto basis functions...")
        start_total_time = time.time()

    l_list = numerics_params['l_list']
    n_list = numerics_params['n_list']

    # Extract parameters
    basis = numerics_params.get('basis', 'haar')
    n_a = numerics_params['n_a']
    n_b = numerics_params['n_b']
    n_c = numerics_params['n_c']
    power_a = numerics_params.get('power_a', 1)
    power_b = numerics_params.get('power_b', 1)
    power_c = numerics_params.get('power_c', 1)
    special_mesh = numerics_params.get('special_mesh', False)

    assert basis in ['haar'], "Projection: Unsupported basis type."
    assert power_b == 1, "Projection: Only power_theta=1 is supported for spherical Haar wavelets."
    assert power_c == 1, "Projection: Only power_phi=1 is supported for spherical Haar wavelets."

    energy_threshold = physics_params['threshold']
    energy_bin_width = numerics_params['energy_bin_width']
    energy_max_factor = numerics_params.get('energy_max_factor', 4.0)
    energy_max = physics.get_energy_max(phonon_file, factor=energy_max_factor)
    energy_bin_num = int((energy_max - energy_threshold)/energy_bin_width) + 1

    q_max = physics_params['q_max']
    q_cut_option = numerics_params['q_cut']
    q_max = physics.get_q_max(q_max=q_max, q_cut_option=q_cut_option,
                              phonon_file=phonon_file,
                              atom_masses=phonopy_params['atom_masses'],
                              verbose=verbose)

    modelname = file_params['modelname']
    csvname = file_params['csvname']
    write_info = {
        'basis': basis,
        'threshold': energy_threshold,
        'energy_bin_width': energy_bin_width,
        'q_max': q_max,
        'n_a': n_a,
        'n_b': n_b,
        'n_c': n_c,
        'power_a': power_a,
        'power_b': power_b,
        'power_c': power_c,
        'special_mesh': special_mesh,
    }

    if verbose:
        print(f"    Using {basis} basis with n_max={max(n_list)}, l_max={max(l_list)}")
        print(f"    Grid size: n_a={n_a}, n_b={n_b}, n_c={n_c}")
        print(
            f"    Grid rescaled: power_a={power_a}, power_b={power_b}, power_c={power_c}")
        print(
            f"    Energy bins: {energy_bin_num} bins from {energy_threshold} eV to {energy_max} eV")

        print("\n    Generate grids and calculating form factor...")
        start_time = time.time()

    # Prepare grid points and basis function values
    lm_list = [(l, m) for l in l_list for m in range(-l, l+1)]
    q_xyz_list, y_lm_vals, jacob_vals = generate_mesh_ylm_jacob(
        lm_list, q_max, n_a, n_b, n_c,
        power_a, power_b, power_c
    )
    if special_mesh:
        q_min = energy_threshold / (const.VESC + const.VE)
        lam_start = basis_funcs.haar_n_to_lam_mu(max(n_list))[0] + 1
        q_xyz_list_exp, jacob_vals_exp, n_list_exp = generate_special_mesh_jacob_nlist(
            q_max, q_min, n_b, n_c, lam_start, verbose=verbose
        )

    # Calculate form factor on grid
    form_factor_bin_vals = physics.form_factor(
        q_xyz_list,
        energy_threshold, energy_bin_width, energy_max,
        numerics_params, phonopy_params, c_dict, phonon_file
    ).reshape(energy_bin_num, n_a, n_b, n_c)
    del q_xyz_list
    if special_mesh:
        form_factor_bin_vals_exp = physics.form_factor(
            q_xyz_list_exp,
            energy_threshold, energy_bin_width, energy_max,
            numerics_params, phonopy_params, c_dict, phonon_file
        ).reshape(energy_bin_num, -1, n_b, n_c)
        del q_xyz_list_exp

    if verbose:
        end_time = time.time()
        print(
            f"    Form factor calculation completed in {end_time - start_time:.2f} seconds.")
        print("\n    Projecting form factor onto basis functions and saving results...")

    # Project form factor onto basis functions
    for i_bin in range(energy_bin_num):

        if verbose:
            if i_bin % (energy_bin_num // 5 + 1) == 0:
                print(
                    f"      Projecting energy bin {i_bin}/{energy_bin_num-1}...")

        f_nlm = proj_get_f_nlm(
            n_list, lm_list,
            form_factor_bin_vals[i_bin, :, :, :], y_lm_vals, jacob_vals,
            basis, n_a, power_a, verbose=verbose
        )
        utility.writeFnlm_csv(csvsave_name=csvname+'_bin_'+str(i_bin)+'.csv',
                              f_nlm_coeffs=f_nlm,
                              info=write_info,
                              use_gvar=False)

        if special_mesh:
            f_nlm_exp = proj_get_f_nlm(
                n_list_exp, lm_list,
                form_factor_bin_vals_exp[i_bin, :,
                                         :, :], y_lm_vals, jacob_vals_exp,
                basis, len(n_list_exp), power_a, special_mesh=True, verbose=verbose
            )
            utility.writeFnlm_csv(csvsave_name=csvname+'_bin_'+str(i_bin)+'.csv',
                                  f_nlm_coeffs=f_nlm_exp,
                                  use_gvar=False)

    if verbose:
        print("    Projection completed for all energy bins.")
        print(f"\n    Coefficients saved to {csvname}_bin_*.csv")
        end_total_time = time.time()
        print(
            f"    Total projection time: {end_total_time - start_total_time:.2f} seconds.")


def proj_vdf(physics_params, numerics_params, file_params,
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

    l_list = numerics_params['l_list']
    n_list = numerics_params['n_list']

    basis = numerics_params.get('basis', 'haar')
    n_a = numerics_params['n_a']
    n_b = numerics_params['n_b']
    n_c = numerics_params['n_c']
    power_a = numerics_params['power_a']
    power_b = numerics_params.get('power_b', 1)
    power_c = numerics_params.get('power_c', 1)

    assert basis in ['haar'], "Projection: Unsupported basis type."
    assert power_b == 1, "Projection: Only power_b=1 is supported for spherical Haar wavelets."
    assert power_c == 1, "Projection: Only power_c=1 is supported for spherical Haar wavelets."
    v_max = physics_params['v_max']
    vdf = physics_params['vdf']
    vdf_params = physics_params['vdf_params']

    modelname = file_params['vdf_model']
    csvname = file_params['csvname']
    write_info = {
        'basis': basis,
        'v_max': v_max,
        'n_a': n_a,
        'n_b': n_b,
        'n_c': n_c,
        'power_a': power_a,
        'power_b': power_b,
        'power_c': power_c,
        'vdf_model': modelname,
        'vdf_params': vdf_params,
    }

    if verbose:
        print(f"    Using {basis} basis with n_max={max(n_list)}, l_max={max(l_list)}")
        print(f"    Grid size: n_a={n_a}, n_b={n_b}, n_c={n_c}")
        print(
            f"    Grid rescaled: power_a={power_a}, power_b={power_b}, power_c={power_c}")

        print("\n    Generate grids and calculating form factor...")
        start_time = time.time()

    # Prepare grid points and basis function values
    lm_list = [(l, m) for l in l_list for m in range(-l, l+1)]
    v_xyz_list, y_lm_vals, jacob_vals = generate_mesh_ylm_jacob(
        lm_list, v_max, n_a, n_b, n_c, power_a, power_b, power_c
    )

    # Calculate vdf on grid
    vdf_vals = np.array(
        [vdf(v_vec, *vdf_params) for v_vec in v_xyz_list]
    ).reshape(n_a, n_b, n_c)
    del v_xyz_list

    if verbose:
        end_time = time.time()
        print(
            f"    VDF calculation completed in {end_time - start_time:.2f} seconds.")
        print("\n    Projecting VDF onto basis functions and saving results...")

    # Project vdf onto basis functions
    f_nlm = proj_get_f_nlm(
        n_list, lm_list,
        vdf_vals, y_lm_vals, jacob_vals,
        basis, n_a, power_a, verbose=verbose
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
        print(
            f"    Total projection time: {end_total_time - start_total_time:.2f} seconds.")


def proj_mcalI(physics_params, numerics_params, file_params,
               verbose=False) -> None:
    """
    Project the McalI integral onto basis functions.

    Args:
        l_list: 
            list[int]: List of angular quantum numbers.
        nv_list: 
            list[int]: List of radial quantum numbers for velocity.
        nq_list: 
            list[int]: List of radial quantum numbers for momentum transfer.
        mass: 
            float: Dark matter mass in eV.
        energy: 
            float: Energy transfer in eV.
        fn: 
            float: Form factor index.
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
        print("\n    Starting projection of McalI onto basis functions...")
        start_total_time = time.time()

    l_list = numerics_params['l_list']
    nv_list = numerics_params['nv_list']
    nq_list = numerics_params['nq_list']
    shape = (len(l_list), len(nv_list), len(nq_list))

    mass_list = physics_params['mass_list']
    fdm_list = physics_params['fdm_list']

    basis = numerics_params.get('basis', 'haar')
    assert basis in ['haar'], "Projection: Unsupported basis type."

    v_max = physics_params['v_max']
    basis_v = dict(u0=v_max, type='wavelet', uMax=v_max)

    q_max_list = physics_params['q_max']
    assert len(mass_list) == len(
        q_max_list), "Length of mass_list and q_max_list must be the same."

    energy_threshold = physics_params['threshold']
    energy_bin_width = numerics_params['energy_bin_width']
    energy_max = numerics_params['energy_max']
    energy_bin_num = int((energy_max - energy_threshold)/energy_bin_width) + 1

    mass_sm = physics_params['mass_sm']

    for i_mass, mass in enumerate(mass_list):

        if verbose:
            print(f"\n    Projecting for mass mX = {mass} eV...")

        q_max = q_max_list[i_mass]
        basis_q = dict(u0=q_max, type='wavelet', uMax=q_max)

        for fdm in fdm_list:

            if verbose:
                print(f"      Projecting for f_n = {fdm}...")

            for i_bin in range(energy_bin_num):

                if verbose:
                    if i_bin % (energy_bin_num // 5 + 1) == 0:
                        print(
                            f"        Projecting energy bin {i_bin}/{energy_bin_num-1}...")

                energy = energy_threshold + (i_bin + 0.5)*energy_bin_width
                dm_model = dict(mX=mass, fdm=fdm, mSM=mass_sm, DeltaE=energy)

                mI = vsdm.McalI(basis_v, basis_q, dm_model,
                                use_gvar=False, do_mcalI=False)

                coef = np.zeros(shape, dtype=float)
                for i_l, l in enumerate(l_list):
                    for i_nv, nv in enumerate(nv_list):
                        for i_nq, nq in enumerate(nq_list):
                            coef[i_l, i_nv, i_nq] = mI.getI_lvq_analytic((l, nv, nq))

                utility.write_hdf5(hdf5file=file_params['hdf5name']+'.hdf5',
                                   groupname=f'mcalI/{mass:.3e}/{fdm}/{i_bin}',
                                   datasetname='Ilvq_mean',
                                   data=coef)
            
            utility.write_hdf5(hdf5file=file_params['hdf5name']+'.hdf5',
                               groupname=f'mcalI/{mass:.3e}/{fdm}/info',
                               info={
                                   'l_list': np.array(l_list),
                                   'nv_list': np.array(nv_list),
                                   'nq_list': np.array(nq_list),
                                   'v_max': np.array([v_max]),
                                   'q_max': np.array([q_max]),
                                   'mass': np.array([mass]),
                                   'fdm': np.array([fdm]),
                                   'mass_sm': np.array([mass_sm]),
                                   'energy_threshold': np.array([energy_threshold]),
                                   'energy_bin_width': np.array([energy_bin_width]),
                                   'energy_bin_num': np.array([energy_bin_num]),
                               })

    if verbose:
        end_total_time = time.time()
        print(
            f"    Total projection time: {end_total_time - start_total_time:.2f} seconds.")
