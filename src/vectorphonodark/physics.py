import numba
import numpy as np
import math
import os
import phonopy

from . import utility
from . import constants as const


def create_vE_vec(t):
    """
    Returns vE_vec for time t given in hours.

    Parameters
    ----------
    t : float
        Time in hours.
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

    Parameters
    ----------
    phonon_file : Phonopy object
        The phonopy object containing the phonon data.
    factor : float, optional
        Multiplicative factor to scale the maximum energy (default is 1.2).

    Returns
    -------
    float
        The maximum phonon energy in eV.
    """
    _, ph_omega_delta_E = run_phonopy(
        phonon_file, [[0., 0., 0.]])

    max_delta_E = factor*np.amax(ph_omega_delta_E)

    return max_delta_E


def compute_q_cut(phonon_file, atom_masses, factor=10.0) -> float:
    """
    Compute the cutoff momentum due to suppression from Debye-Waller factor.

    Parameters
    ----------
    phonon_file : Phonopy object
        The phonopy object containing the phonon data.
    atom_masses : array-like
        The masses of the atoms.
    factor : float, optional
        Multiplicative factor to scale the q cut (default is 10.0).

    Returns
    -------
    float
        The computed q cut value.
    """

    _, ph_omega = run_phonopy(
        phonon_file, np.array([[0., 0., 0.]]))

    q_cut = factor*np.sqrt(np.amax(atom_masses)*np.amax(ph_omega))

    return q_cut


def run_phonopy(phonon_file, k_mesh):
    """
    Given a phonon file and k mesh, Returns eigenvectors and frequencies in eV

    Parameters
    ----------
    phonon_file : Phonopy object
        The phonopy object containing the phonon data.
    k_mesh : array-like
        The k-point mesh.

    Returns
    -------
    tuple
        A tuple containing eigenvectors and frequencies in eV.
    """

    # run phonopy in mesh mode 
    phonon_file.run_qpoints(k_mesh, with_eigenvectors=True)

    n_k = len(k_mesh)

    mesh_dict = phonon_file.get_qpoints_dict()

    eigenvectors_pre = mesh_dict['eigenvectors']

    # convert frequencies to correct units
    omega = 2*const.PI*(const.THz_To_eV)*mesh_dict['frequencies']

    num_atoms = len(phonon_file.primitive)
    num_modes = 3*num_atoms 

    # q, nu, i, alpha
    eigenvectors = np.zeros((n_k, num_modes, num_atoms, 3), dtype=complex)

    # sort the eigenvectors
    for q in range(n_k):
        for nu in range(num_modes):
            eigenvectors[q][nu][:][:] = np.array_split(
                    eigenvectors_pre[q].T[nu], num_atoms)

    return [eigenvectors, omega]


# def load_phonopy_file(material, io_parameters, supercell, poscar_path, force_sets_path, born_path,
#                         proc_id = 0, root_process = 0):


#     if os.path.exists(io_parameters['material_data_folder']+material+'/BORN'):

#         born_exists = True

#     else:

#         if proc_id == root_process: 

#             print('\tThere is no BORN file for '+material)
#             print()

#         born_exists = False

#     if born_exists: 

#         phonon_file = phonopy.load(
#                             supercell_matrix    = supercell,
#                             primitive_matrix    = 'auto',
#                             unitcell_filename   = poscar_path,
#                             force_sets_filename = force_sets_path,
#                             is_nac              = True,
#                             born_filename       = born_path
#                            )

#     else:

#         if proc_id == root_process:

#             print('\tNo BORN file found for : '+material)

#         raise NotImplementedError("Phonopy utility: Loading without BORN file is not implemented.")
    
#         # phonon_file = phonopy.load(
#         #                     supercell_matrix    = supercell_data[material],
#         #                     primitive_matrix    = 'auto',
#         #                     unitcell_filename   = poscar_path,
#         #                     force_sets_filename = force_sets_path
#         #                    )

#     return [phonon_file, born_exists]


def get_phonon_file_data(phonon_file, born_exists):
    """
    Returns
    -------
    dict with phonopy parameters:

        n_atoms - number of atoms in primitive cell
        n_modes - number of modes = 3*n_atoms

        Transformation matrices:
        pos_red_to_XYZ - reduced coordinate positions to XYZ
        pos_XYZ_to_red - XYZ coordinates to red
        recip_red_to_XYZ - reduced coordinates to XYZ
        recip_XYZ_to_red - XYZ coordinates to reduced

        eq_positions - equilibrium positions of atoms
        atom_masses - masses of atoms in eV
        A_list - Mass numbers (A)
        Z_list - atomic numbers (Z)
        born - Z_j
        dielectric - high frequency dielectric
    """

    num_atoms = len(phonon_file.primitive)
    num_modes = 3*num_atoms 

    A_list = phonon_file.primitive.masses
    Z_list = phonon_file.primitive.numbers

    eq_positions_XYZ = const.Ang_To_inveV*phonon_file.primitive.positions

    atom_masses = const.AMU_To_eV*phonon_file.primitive.masses

    primitive_mat = phonon_file.primitive.cell

    pos_red_to_XYZ = const.Ang_To_inveV*np.transpose(primitive_mat)
    pos_XYZ_to_red = np.linalg.inv(pos_red_to_XYZ)

    a_vec = np.matmul(pos_red_to_XYZ, [1, 0, 0])
    b_vec = np.matmul(pos_red_to_XYZ, [0, 1, 0])
    c_vec = np.matmul(pos_red_to_XYZ, [0, 0, 1])

    recip_lat_a = 2*const.PI*(np.cross(b_vec, c_vec))/(np.matmul(a_vec, np.cross(b_vec, c_vec)))
    recip_lat_b = 2*const.PI*(np.cross(c_vec, a_vec))/(np.matmul(b_vec, np.cross(c_vec, a_vec)))
    recip_lat_c = 2*const.PI*(np.cross(a_vec, b_vec))/(np.matmul(c_vec, np.cross(a_vec, b_vec)))

    recip_red_to_XYZ = np.transpose([recip_lat_a, recip_lat_b, recip_lat_c])
    recip_XYZ_to_red = np.linalg.inv(recip_red_to_XYZ)

    if born_exists:

        born       = phonon_file.nac_params['born']
        dielectric = phonon_file.nac_params['dielectric']

    else:

        born       = np.zeros((num_atoms, 3, 3))
        dielectric = np.identity(3)

    return {
        'num_atoms': num_atoms,
        'num_modes': num_modes,
        'pos_red_to_XYZ': pos_red_to_XYZ,
        'pos_XYZ_to_red': pos_XYZ_to_red,
        'recip_red_to_XYZ': recip_red_to_XYZ,
        'recip_XYZ_to_red': recip_XYZ_to_red,
        'eq_positions_XYZ': eq_positions_XYZ,
        'atom_masses': atom_masses,
        'A_list': A_list,
        'Z_list': Z_list,
        'born': born,
        'dielectric': dielectric
    }


def get_material_data(material_input, physics_model_input, numerics_input):
    """
    Given material, physics model, and numerics input file paths,
    returns phonon file, phonopy parameters, c_dict, and n_DW_params.
    """

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

    phonopy_params = get_phonon_file_data(phonon_file, born_exists)

    n_DW_params = {
        "n_DW_x": num_mod.numerics_parameters["n_DW_x"],
        "n_DW_y": num_mod.numerics_parameters["n_DW_y"],
        "n_DW_z": num_mod.numerics_parameters["n_DW_z"],
    }

    return phonon_file, phonopy_params, c_dict, n_DW_params


@numba.njit
def get_kG_from_q_red(q_red_vec, q_red_to_XYZ):
    """
    Parameters
    ----------
    q_red_vec : 
        q vector in reduced coordinates
    q_red_to_XYZ : 
        matrix converting q in reduced coordinates to XYZ

    Returns
    -------
    output : 
        [k_red_vec, G_red_vec]: the k and G vectors in reduced coordinates

    """
    set_of_closest_G_red = np.zeros((8, 3), dtype=np.float64)

    set_of_closest_G_red[0] = [
        math.floor(q_red_vec[0]),
        math.floor(q_red_vec[1]),
        math.floor(q_red_vec[2]),
    ]
    set_of_closest_G_red[1] = [
        math.floor(q_red_vec[0]),
        math.floor(q_red_vec[1]),
        math.ceil(q_red_vec[2]),
    ]
    set_of_closest_G_red[2] = [
        math.floor(q_red_vec[0]),
        math.ceil(q_red_vec[1]),
        math.floor(q_red_vec[2]),
    ]
    set_of_closest_G_red[3] = [
        math.ceil(q_red_vec[0]),
        math.floor(q_red_vec[1]),
        math.floor(q_red_vec[2]),
    ]
    set_of_closest_G_red[4] = [
        math.floor(q_red_vec[0]),
        math.ceil(q_red_vec[1]),
        math.ceil(q_red_vec[2]),
    ]
    set_of_closest_G_red[5] = [
        math.ceil(q_red_vec[0]),
        math.floor(q_red_vec[1]),
        math.ceil(q_red_vec[2]),
    ]
    set_of_closest_G_red[6] = [
        math.ceil(q_red_vec[0]),
        math.ceil(q_red_vec[1]),
        math.floor(q_red_vec[2]),
    ]
    set_of_closest_G_red[7] = [
        math.ceil(q_red_vec[0]),
        math.ceil(q_red_vec[1]),
        math.ceil(q_red_vec[2]),
    ]

    # q_XYZ_vec = q_red_to_XYZ @ q_red_vec
    q_XYZ_vec = [
        q_red_to_XYZ[0, 0] * q_red_vec[0]
        + q_red_to_XYZ[0, 1] * q_red_vec[1]
        + q_red_to_XYZ[0, 2] * q_red_vec[2],
        q_red_to_XYZ[1, 0] * q_red_vec[0]
        + q_red_to_XYZ[1, 1] * q_red_vec[1]
        + q_red_to_XYZ[1, 2] * q_red_vec[2],
        q_red_to_XYZ[2, 0] * q_red_vec[0]
        + q_red_to_XYZ[2, 1] * q_red_vec[1]
        + q_red_to_XYZ[2, 2] * q_red_vec[2],
    ]

    first = True

    for vec in set_of_closest_G_red:

        # diff_vec = q_XYZ_vec - q_red_to_XYZ @ vec
        diff_vec = [
            q_XYZ_vec[0]
            - (
                q_red_to_XYZ[0, 0] * vec[0]
                + q_red_to_XYZ[0, 1] * vec[1]
                + q_red_to_XYZ[0, 2] * vec[2]
            ),
            q_XYZ_vec[1]
            - (
                q_red_to_XYZ[1, 0] * vec[0]
                + q_red_to_XYZ[1, 1] * vec[1]
                + q_red_to_XYZ[1, 2] * vec[2]
            ),
            q_XYZ_vec[2]
            - (
                q_red_to_XYZ[2, 0] * vec[0]
                + q_red_to_XYZ[2, 1] * vec[1]
                + q_red_to_XYZ[2, 2] * vec[2]
            ),
        ]
        diff_vec_sq = (
            diff_vec[0] * diff_vec[0]
            + diff_vec[1] * diff_vec[1]
            + diff_vec[2] * diff_vec[2]
        )

        if first:
            min_dist_sq = diff_vec_sq
            first = False

        if diff_vec_sq <= min_dist_sq:
            min_vec = vec

    G_red_vec = min_vec

    # k_red_vec = np.array(q_red_vec) - np.array(G_red_vec)
    k_red_vec = [
        q_red_vec[0] - G_red_vec[0],
        q_red_vec[1] - G_red_vec[1],
        q_red_vec[2] - G_red_vec[2],
    ]

    return k_red_vec, G_red_vec


@numba.njit
def get_kG_from_q_XYZ(q_XYZ_vec, q_red_to_XYZ) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the k and G vectors in reduced coordinates given a single q vector.

    Parameters
    ----------
    q_XYZ_vec : np.ndarray
        q vector in XYZ coordinates
    q_red_to_XYZ : np.ndarray
        matrix converting q in reduced coordinates to XYZ

    Returns
    -------
    output : tuple[np.ndarray, np.ndarray]
        [k_red_vec, G_red_vec]: the k and G vectors in reduced coordinates
    """

    q_red_vec = np.dot(np.linalg.inv(q_red_to_XYZ), q_XYZ_vec)

    set_of_closest_G_red = np.zeros((8, 3), dtype=np.float64)

    set_of_closest_G_red[0] = np.array(
        [math.floor(q_red_vec[0]), math.floor(q_red_vec[1]), math.floor(q_red_vec[2])]
    )
    set_of_closest_G_red[1] = np.array(
        [math.floor(q_red_vec[0]), math.floor(q_red_vec[1]), math.ceil(q_red_vec[2])]
    )
    set_of_closest_G_red[2] = np.array(
        [math.floor(q_red_vec[0]), math.ceil(q_red_vec[1]), math.floor(q_red_vec[2])]
    )
    set_of_closest_G_red[3] = np.array(
        [math.ceil(q_red_vec[0]), math.floor(q_red_vec[1]), math.floor(q_red_vec[2])]
    )
    set_of_closest_G_red[4] = np.array(
        [math.floor(q_red_vec[0]), math.ceil(q_red_vec[1]), math.ceil(q_red_vec[2])]
    )
    set_of_closest_G_red[5] = np.array(
        [math.ceil(q_red_vec[0]), math.floor(q_red_vec[1]), math.ceil(q_red_vec[2])]
    )
    set_of_closest_G_red[6] = np.array(
        [math.ceil(q_red_vec[0]), math.ceil(q_red_vec[1]), math.floor(q_red_vec[2])]
    )
    set_of_closest_G_red[7] = np.array(
        [math.ceil(q_red_vec[0]), math.ceil(q_red_vec[1]), math.ceil(q_red_vec[2])]
    )

    first = True

    for vec in set_of_closest_G_red:

        vec = np.ascontiguousarray(vec)
        diff_vec = q_XYZ_vec - q_red_to_XYZ @ vec

        if first:
            min_dist_sq = np.dot(diff_vec, diff_vec)
            min_vec = vec
            first = False

        if np.dot(diff_vec, diff_vec) <= min_dist_sq:
            min_dist_sq = np.dot(diff_vec, diff_vec)
            min_vec = vec

    G_red_vec = min_vec

    k_red_vec = q_red_vec - G_red_vec

    return k_red_vec, G_red_vec


@numba.njit
def get_kG_list_from_q_XYZ_list(
    q_XYZ_list, recip_red_to_XYZ
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the k and G vectors in reduced coordinates given a list of q vectors.

    Parameters
    ----------
    q_XYZ_list : np.ndarray
        List of q vectors in XYZ coordinates
    recip_red_to_XYZ : np.ndarray
        matrix converting q in reduced coordinates to XYZ

    Returns
    -------
    output : tuple[np.ndarray, np.ndarray]
        k_mesh: np.ndarray of k vectors in reduced coordinates
        G_xyz_list: np.ndarray of G vectors in XYZ coordinates
    """

    n_q = len(q_XYZ_list)

    k_mesh = np.zeros((n_q, 3), dtype=np.float64)
    G_xyz_list = np.zeros((n_q, 3), dtype=np.float64)

    for i_q in numba.prange(n_q):

        q_XYZ = q_XYZ_list[i_q]

        k_red_vec, G_red_vec = get_kG_from_q_XYZ(q_XYZ, recip_red_to_XYZ)
        k_mesh[i_q] = k_red_vec
        G_xyz_list[i_q] = recip_red_to_XYZ @ G_red_vec

    return k_mesh, G_xyz_list


def get_G_eigenvectors_omega_from_q_XYZ(
    q_XYZ_list, phonon_file, phonopy_params
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get q_XYZ, G_XYZ, eigenvectors, and omega from a q_sph mesh.

    Parameters
    ----------
    q_XYZ_list : np.ndarray
        List of q vectors in XYZ coordinates
    phonon_file : Phonopy object
        The phonopy object containing the phonon data.
    phonopy_params : dict
        The phonopy parameters including transformation matrices.

    Returns
    -------
    tuple
        G_XYZ_list: np.ndarray of G vectors in XYZ coordinates
        ph_eigenvectors: np.ndarray of phonon eigenvectors
        ph_omega: np.ndarray of phonon frequencies in eV
    """

    # reciprocal matrix to convert q in reduced coordinates to XYZ
    recip_red_to_XYZ = np.array(phonopy_params["recip_red_to_XYZ"])

    # get corresponding vector k in the first Brillouin zone and G = q - k
    k_red_list, G_XYZ_list = get_kG_list_from_q_XYZ_list(q_XYZ_list, recip_red_to_XYZ)

    # run phonopy to get polarization vectors and photon energies
    [ph_eigenvectors, ph_omega] = run_phonopy(phonon_file, k_red_list)

    return G_XYZ_list, ph_eigenvectors, ph_omega


# def get_q_max(q_max, q_cut_option=False, 
#               phonon_file=None, atom_masses=None, 
#               verbose=True):
#     """
#     Returns q_max based on input and Debye-Waller factor.

#     Parameters
#     ----------
#     q_max : float
#         The initial maximum momentum transfer.
#     q_cut_option : bool, optional
#         Whether to compute q_cut based on Debye-Waller factor (default is False).
#     phonon_file : Phonopy object, optional
#         The phonopy object containing the phonon data (required if q_cut_option is True).
#     atom_masses : array-like, optional
#         The masses of the atoms (required if q_cut_option is True).
#     verbose : bool, optional
#         Whether to print verbose output (default is True).

#     Returns
#     -------
#     float
#         The adjusted maximum momentum transfer q_max.
#     """

#     if q_cut_option:
#         q_cut = compute_q_cut(phonon_file, atom_masses)
#         if q_cut <= q_max:
#             q_max = q_cut
#             if verbose:
#                 print(f"    Adjusted q_max to {q_max:.4f} eV due to Debye Waller factor.")
#         else:
#             if verbose:
#                 print(f"    Using specified q_max = {q_max:.4f} eV.")
#     else:
#         if verbose:
#             print(f"    Using specified q_max = {q_max:.4f} eV.")

#     return q_max


def get_q_max(material_input: str, factor: float = 10.0):
    """
    Get the maximum q value based on the Debye-Waller factor for the given material.

    Parameters
    ----------
    material_input : str
        The file path to the material input file.
    factor : float, optional
        The multiplicative factor to scale the q cut (default is 10.0).

    Returns
    -------
    float
        The computed maximum q value.
    """

    mat_input_mod_name = os.path.splitext(os.path.basename(material_input))[0]
    mat_mod = utility.import_file(mat_input_mod_name, os.path.join(material_input))
    material = mat_mod.material

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

    phonopy_params = get_phonon_file_data(phonon_file, born_exists)

    return compute_q_cut(phonon_file, phonopy_params["atom_masses"], factor=factor)


def calculate_W_tensor(phonon_file, num_atoms, atom_masses,
                       n_k_1, n_k_2, n_k_3, q_red_to_XYZ):
    """
    Calculate the W tensor, 
    which is related to the Debye-Waller factor by q.W.q = DW_factor, 
    based on a Monkhort-Pack mesh 
    with total number of k points < n_k_1*n_k_2*n_k_3

    Returns
    -------
    np.ndarray
        W_tensor of shape (num_atoms, 3, 3)
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

                [k_vec, G_vec] = get_kG_from_q_red(
                    q_red_vec, q_red_to_XYZ)

                if G_vec[0] == 0.0 and G_vec[1] == 0.0 and G_vec[2] == 0.0:
                    n_k_tot += 1

                    k_list.append(k_vec)

    [eigenvectors, omega] = run_phonopy(phonon_file, k_list)

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


def form_factor(q_XYZ_list: np.ndarray,
                energy_threshold: float, energy_bin_width: float, energy_max: float,
                numerics_params, phonopy_params, c_dict, phonon_file) -> np.ndarray:
    """
    Calculate the form factor for given q vectors and energy bins.

    Parameters
    ----------
    q_XYZ_list : np.ndarray
        The list of q vectors in Cartesian coordinates.
    energy_threshold : float
        The energy threshold for phonon excitations.
    energy_bin_width : float
        The width of each energy bin.
    energy_max : float
        The maximum energy difference.
    phonon_file : Phonopy object
        The Phonopy object containing the phonon data.
    phonopy_params : dict
        The phonopy parameters including dielectric matrix, born charges, etc.
    numerics_params: dict
        The numerical parameters including number of k-points in each direction.
    c_dict: dict
        The coupling constants for electrons, protons, and neutrons.

    Returns
    -------
    np.ndarray
        The form factor values of shape (n_bins, n_q).
    """

    # Prepare G, epsilon, omega, W(q)
    G_xyz_list, ph_eigenvectors, ph_omega = get_G_eigenvectors_omega_from_q_XYZ(
        q_XYZ_list, phonon_file, phonopy_params)

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

    form_factor_bin_vals = _form_factor_numba(
        q_XYZ_list, G_xyz_list, ph_eigenvectors, ph_omega,
        energy_threshold, energy_bin_width, energy_max,
        W_tensor, fe0, fn0, fp0,
        dielectric, num_modes, num_atoms, eq_positions_XYZ,
        A_list, Z_list, born, atom_masses
    )

    return form_factor_bin_vals


@numba.njit
def _form_factor_numba(q_xyz_list, G_xyz_list, ph_eigenvectors, ph_omega,
                      energy_threshold, energy_bin_width, energy_max,
                      W_tensor, fe0, fn0, fp0,
                      dielectric, num_modes, num_atoms, eq_positions_XYZ,
                      A_list, Z_list, born, atom_masses) -> np.ndarray:

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
