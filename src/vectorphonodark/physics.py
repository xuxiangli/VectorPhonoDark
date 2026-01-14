import numba
import numpy as np
import math

from . import constants as const
from . import phonopy_funcs
from . import utility


def create_vE_vec(t):
    """
            Returns vE_vec for time t given in hours
    """
    phi = 2*const.PI*(t/24.0)

    vEx = const.VE*np.sin(const.THETA_E)*np.sin(phi)
    vEy = const.VE*np.cos(const.THETA_E)*np.sin(const.THETA_E)*(np.cos(phi)-1)
    vEz = const.VE*(
        (np.sin(const.THETA_E)**2) * np.cos(phi)
        + np.cos(const.THETA_E)**2
    )

    return np.array([vEx, vEy, vEz])


def get_energy_max(phonon_file, factor=1.2) -> float:
    """
    Get the maximum phonon energy from the phonon file.

    Args:
        phonon_file: 
            Phonopy object: The phonopy object containing the phonon data.

    Returns:
        float: The maximum phonon energy in eV.
    """
    _, ph_omega_delta_E = phonopy_funcs.run_phonopy(
        phonon_file, [[0., 0., 0.]])

    max_delta_E = factor*np.amax(ph_omega_delta_E)

    return max_delta_E


def compute_q_cut(phonon_file, atom_masses):
    """
        Returns q = 10*sqrt(max(m)*max(omega))
    """

    _, ph_omega = phonopy_funcs.run_phonopy(
        phonon_file, np.array([[0., 0., 0.]]))

    q_cut = 10.0*np.sqrt(np.amax(atom_masses)*np.amax(ph_omega))

    return q_cut


def get_q_max(q_max, q_cut_option=False, 
              phonon_file=None, atom_masses=None, 
              verbose=True):
    """
        Returns q_max based on input or Debye-Waller factor
    """

    if q_cut_option:
        q_cut = compute_q_cut(phonon_file, atom_masses)
        if q_cut <= q_max:
            q_max = q_cut
            if verbose:
                print(f"    Adjusted q_max to {q_max:.4f} eV due to Debye Waller factor.")
        else:
            if verbose:
                print(f"    Using specified q_max = {q_max:.4f} eV.")
    else:
        if verbose:
            print(f"    Using specified q_max = {q_max:.4f} eV.")

    return q_max


def calculate_W_tensor(phonon_file, num_atoms, atom_masses,
                       n_k_1, n_k_2, n_k_3, q_red_to_XYZ):
    """
        Calculate the W tensor, 
        which is related to the Debye-Waller factor by q.W.q = DW_factor, 
        based on a Monkhort-Pack mesh 
        with total number of k points < n_k_1*n_k_2*n_k_3

        Returns: W_tens(j, a, b)
    """

    n_k_tot = 0
    k_list = []

    for i in range(n_k_1):
        for j in range(n_k_2):
            for k in range(n_k_3):

                q_red_vec = []

                q_red_vec.append((2.0*i - n_k_1 + 1.0)/n_k_1)
                q_red_vec.append((2.0*j - n_k_2 + 1.0)/n_k_2)
                q_red_vec.append((2.0*k - n_k_3 + 1.0)/n_k_3)

                [k_vec, G_vec] = utility.get_kG_from_q_red(
                    q_red_vec, q_red_to_XYZ)

                if G_vec[0] == 0.0 and G_vec[1] == 0.0 and G_vec[2] == 0.0:
                    n_k_tot += 1

                    k_list.append(k_vec)

    [eigenvectors, omega] = phonopy_funcs.run_phonopy(phonon_file, k_list)

    W_tensor = np.zeros((num_atoms, 3, 3), dtype=complex)

    for j in range(num_atoms):
        for k in range(n_k_tot):
            for nu in range(3*num_atoms):
                for a in range(3):
                    for b in range(3):

                        W_tensor[j][a][b] += (
                            (4.0*atom_masses[j]*n_k_tot*omega[k, nu])**(-1) *
                            eigenvectors[k, nu, j, a] *
                            np.conj(eigenvectors[k, nu, j, b])
                        )

    return np.array(W_tensor)


def form_factor(q_xyz_list: np.ndarray,
                energy_threshold: float, energy_bin_width: float, energy_max: float,
                numerics_params, phonopy_params, c_dict, phonon_file) -> np.ndarray:
    """
    Calculate the form factor for given q vector and energy bin info.

    Args:
        q_xyz_list: 
            np.ndarray: The list of q vectors in Cartesian coordinates.
        threshold: 
            float: The energy threshold for phonon excitations.
        energy_bin_width: 
            float: The width of each energy bin.
        max_delta_E: 
            float: The maximum energy difference to consider.
        phonon_file: 
            Phonopy object: The phonopy object containing the phonon data.
        phonopy_params: 
            dict: The phonopy parameters including dielectric matrix, born charges, etc.
        numerics_params: 
            dict: The numerical parameters including number of k-points in each direction.
        c_dict: 
            dict: The coupling constants for electrons, protons, and neutrons.

    Returns:
        np.ndarray: The form factor values for each energy bin.
    """

    # Prepare G, epsilon, omega, W(q)
    G_xyz_list, ph_eigenvectors, ph_omega = utility.get_G_eigenvectors_omega_from_q_xyz(
        q_xyz_list, phonon_file, phonopy_params)

    W_tensor = calculate_W_tensor(phonon_file,
                                  phonopy_params['num_atoms'],
                                  phonopy_params['atom_masses'],
                                  numerics_params['n_DW_x'],
                                  numerics_params['n_DW_y'],
                                  numerics_params['n_DW_z'],
                                  phonopy_params['recip_red_to_XYZ'])

    fe0 = c_dict[1]['e']
    fn0 = c_dict[1]['n']
    fp0 = c_dict[1]['p']

    dielectric = phonopy_params['dielectric']
    atom_masses = phonopy_params['atom_masses']
    num_modes = phonopy_params['num_modes']
    num_atoms = phonopy_params['num_atoms']
    eq_positions_XYZ = phonopy_params['eq_positions_XYZ']

    A_list = phonopy_params['A_list']
    Z_list = phonopy_params['Z_list']
    born = phonopy_params['born']

    form_factor_bin_vals = form_factor_numba(
        q_xyz_list, G_xyz_list, ph_eigenvectors, ph_omega,
        energy_threshold, energy_bin_width, energy_max,
        W_tensor, fe0, fn0, fp0,
        dielectric, num_modes, num_atoms, eq_positions_XYZ,
        A_list, Z_list, born, atom_masses
    )

    return form_factor_bin_vals


@numba.njit(fastmath=False)
def form_factor_numba(q_xyz_list, G_xyz_list, ph_eigenvectors, ph_omega,
                      energy_threshold, energy_bin_width, energy_max,
                      W_tensor, fe0, fn0, fp0,
                      dielectric, num_modes, num_atoms, eq_positions_XYZ,
                      A_list, Z_list, born, atom_masses) -> np.ndarray:
    """
    Calculate the form factor for given q vector and energy bin info.

    Args:
        q_xyz_list: 
            np.ndarray: The list of q vectors in Cartesian coordinates.
        threshold: 
            float: The energy threshold for phonon excitations.
        energy_bin_width: 
            float: The width of each energy bin.
        max_delta_E: 
            float: The maximum energy difference to consider.
        phonon_file: 
            Phonopy object: The phonopy object containing the phonon data.
        phonopy_params: 
            dict: The phonopy parameters including dielectric matrix, born charges, etc.
        numerics_params: 
            dict: The numerical parameters including number of k-points in each direction.
        c_dict: 
            dict: The coupling constants for electrons, protons, and neutrons.

    Returns:
        np.ndarray: The form factor values for each energy bin.
    """

    m_cell = np.sum(atom_masses)

    n_bin = math.floor((energy_max-energy_threshold)/energy_bin_width) + 1

    n_q = len(q_xyz_list)

    form_factor_bin_vals = np.zeros((n_bin, n_q), dtype=np.float64)

    for i_q in range(n_q):

        q_xyz = q_xyz_list[i_q]
        q0, q1, q2 = q_xyz[0], q_xyz[1], q_xyz[2]
        q_hat = q_xyz / np.sqrt(q0*q0 + q1*q1 + q2*q2)

        screen_val = 1.0/np.dot(q_hat, dielectric @ q_hat)

        # Eq 50 in 1910.08092
        fe = screen_val*fe0
        fp = fp0 + (1.0 - screen_val)*fe0
        fn = fn0

        for nu in range(num_modes):

            energy_diff = ph_omega[i_q][nu]

            if energy_diff >= energy_threshold:

                i_bin = math.floor(
                    (energy_diff-energy_threshold)/energy_bin_width)

                S_nu = 0.0 + 0.0j

                for j in range(num_atoms):

                    # dw_val_j = np.dot(q_xyz, W_tensor[j] @ q_xyz)
                    Wj = W_tensor[j]
                    dw_val_j = (
                        q0*(Wj[0, 0]*q0 + Wj[0, 1]*q1 + Wj[0, 2]*q2)
                        + q1*(Wj[1, 0]*q0 + Wj[1, 1]*q1 + Wj[1, 2]*q2)
                        + q2*(Wj[2, 0]*q0 + Wj[2, 1]*q1 + Wj[2, 2]*q2)
                    )

                    pos_phase_j = (
                        1j)*np.dot(G_xyz_list[i_q], eq_positions_XYZ[j])

                    A_j = A_list[j]
                    Z_j = Z_list[j]

                    Y_j = (-fe*(born[j] @ q_xyz)
                           + fe*Z_j*q_xyz
                           + fn*(A_j - Z_j)*q_xyz
                           + fp*Z_j*q_xyz)

                    # Y_dot_e_star = np.dot(Y_j, np.conj(ph_eigenvectors[i_q][nu][j]))
                    Y_dot_e_star = (
                        Y_j[0] * np.conj(ph_eigenvectors[i_q, nu, j, 0])
                        + Y_j[1] * np.conj(ph_eigenvectors[i_q, nu, j, 1])
                        + Y_j[2] * np.conj(ph_eigenvectors[i_q, nu, j, 2])
                    )

                    S_nu += ((atom_masses[j])**(-0.5) *
                             np.exp(-dw_val_j + pos_phase_j)*Y_dot_e_star)

                form_factor_bin_vals[i_bin, i_q] += np.real(
                    0.5*(1.0/m_cell)*(1.0/energy_diff)*S_nu*np.conj(S_nu)
                )

    return form_factor_bin_vals
