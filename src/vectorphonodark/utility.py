import numpy as np
import math
import numba
import sys
import os
import phonopy
import quaternionic
from importlib import util
from functools import reduce

from . import constants as const
# from . import phonopy_funcs


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

    Args:
        full_name:
            str: The full name to assign to the module.
        path:
            str: The file path to the module.

    Returns:
        module: The imported module.
    """

    spec = util.spec_from_file_location(full_name, path)
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


@numba.njit
def sph_to_cart(vec_sph) -> np.ndarray:
    """
    Convert spherical coordinates to Cartesian coordinates.

    Args:
        vec_sph:
            np.ndarray: An array of shape (..., 3) representing points in spherical coordinates (r, theta, phi).

    Returns:
        np.ndarray: An array of shape (..., 3) representing points in Cartesian coordinates (x, y, z).
    """
    r = vec_sph[..., 0]
    theta = vec_sph[..., 1]
    phi = vec_sph[..., 2]

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.stack((x, y, z), axis=-1)


def get_intersection_index(base=None, **lists):
    """
    Get the indices of common elements in multiple lists.

    Args:
        base_list (list): The base list to compare against.
        **lists: Arbitrary number of lists to find common elements with.

    Returns:
        np.ndarray: Array of indices in each list corresponding to the common elements.
            shape: (number of lists, number of common elements)
    """
    arrays = [np.asarray(l) for l in lists.values()]
    base_arr = np.asarray(base) if base is not None else np.array([])
    if base_arr.size > 0:
        arrays.insert(0, base_arr)

    if not arrays:
        return np.array([])

    common_elements = reduce(np.intersect1d, arrays)

    if common_elements.size == 0:
        return np.empty((len(lists), 0), dtype=int)

    target_arrays = [np.asarray(l) for l in lists.values()]
    indices = np.vstack(
        [np.searchsorted(arr, common_elements) for arr in target_arrays]
    )
    return indices


def getQ(theta, phi):
    axisphi = phi + np.pi / 2  # stationary under R
    axR = theta / 2
    qr = np.cos(axR)
    qi = np.sin(axR) * np.cos(axisphi)
    qj = np.sin(axR) * np.sin(axisphi)
    qk = 0.0
    return quaternionic.array(qr, qi, qj, qk)


# def run_phonopy(phonon_file, k_red) -> list[np.ndarray]:
#     """
#     Run phonopy to compute phonon eigenvectors and frequencies at given k-points.

#     Args:
#         phonon_file:
#             Phonopy object: The phonopy object containing the phonon data.
#         k_red:
#             np.ndarray: An array of shape (N, 3) representing the reduced k-points.

#     Returns:
#         list[np.ndarray]: A list containing two elements:
#             - eigenvectors: An array of shape (num_modes, num_atoms, 3) representing the phonon eigenvectors.
#             - omega: An array of shape (num_modes,) representing the phonon frequencies in eV.
#     """

#     # run phonopy in mesh mode
#     phonon_file.run_qpoints(k_red, with_eigenvectors=True)

#     mesh_dict = phonon_file.get_qpoints_dict()

#     eigenvectors_pre = mesh_dict["eigenvectors"]

#     # convert frequencies to correct units
#     omega = 2 * const.PI * (const.THz_To_eV) * mesh_dict["frequencies"][0]

#     num_atoms = phonon_file.primitive.get_number_of_atoms()
#     num_modes = 3 * num_atoms

#     # q, nu, i, alpha
#     eigenvectors = np.zeros((num_modes, num_atoms, 3), dtype=complex)

#     # sort the eigenvectors
#     for nu in range(num_modes):
#         eigenvectors[nu][:][:] = np.array(
#             np.array_split(eigenvectors_pre.T[nu], num_atoms)
#         ).reshape(num_atoms, 3)

#     return [eigenvectors, omega]


def run_phonopy(phonon_file, k_mesh):
    """
        Given a phonon file and k mesh, Returns eigenvectors and frequencies in eV
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


def load_phonopy_file(material, io_parameters, supercell, poscar_path, force_sets_path, born_path,
                        proc_id = 0, root_process = 0):


    if os.path.exists(io_parameters['material_data_folder']+material+'/BORN'):

        born_exists = True

    else:

        if proc_id == root_process: 

            print('\tThere is no BORN file for '+material)
            print()

        born_exists = False

    if born_exists: 

        phonon_file = phonopy.load(
                            supercell_matrix    = supercell,
                            primitive_matrix    = 'auto',
                            unitcell_filename   = poscar_path,
                            force_sets_filename = force_sets_path,
                            is_nac              = True,
                            born_filename       = born_path
                           )

    else:

        if proc_id == root_process:

            print('\tNo BORN file found for : '+material)

        raise NotImplementedError("Phonopy utility: Loading without BORN file is not implemented.")
    
        # phonon_file = phonopy.load(
        #                     supercell_matrix    = supercell_data[material],
        #                     primitive_matrix    = 'auto',
        #                     unitcell_filename   = poscar_path,
        #                     force_sets_filename = force_sets_path
        #                    )

    return [phonon_file, born_exists]


def get_phonon_file_data(phonon_file, born_exists):
    """
        Returns:

            n_atoms - number of atoms in primitive cell

            n_modes - number of modes = 3*n_atoms

            Transformation matrices

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
            'dielectric': dielectric}


@numba.njit
def get_kG_from_q_red(q_red_vec, q_red_to_XYZ):
    """
    q_red_vec: q vector in reduced coordinates
    q_red_to_XYZ: matrix converting q in reduced coordinates to XYZ

    output: [k_red_vec, G_red_vec]: the k and G vectors in reduced coordinates

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
    q_XYZ_vec: q vector in XYZ coordinates
    q_red_to_XYZ: matrix converting q in reduced coordinates to XYZ

    output: [k_red_vec, G_red_vec]: the k and G vectors in reduced coordinates
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
def get_kG_list_from_q_xyz_list(
    q_xyz_list, recip_red_to_XYZ
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the k and G vectors in reduced coordinates given a q vector.
    """

    n_q = len(q_xyz_list)

    k_mesh = np.zeros((n_q, 3), dtype=np.float64)
    G_xyz_list = np.zeros((n_q, 3), dtype=np.float64)

    for i_q in range(n_q):

        q_xyz = q_xyz_list[i_q]

        k_red_vec, G_red_vec = get_kG_from_q_XYZ(q_xyz, recip_red_to_XYZ)
        k_mesh[i_q] = k_red_vec
        G_xyz_list[i_q] = recip_red_to_XYZ @ G_red_vec

    return k_mesh, G_xyz_list


def get_G_eigenvectors_omega_from_q_xyz(
    q_xyz_list, phonon_file, phonopy_params
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get q_xyz, G_xyz, eigenvectors, and omega from a q_sph mesh.
    """

    # reciprocal matrix to convert q in reduced coordinates to XYZ
    recip_red_to_XYZ = np.array(phonopy_params["recip_red_to_XYZ"])

    # get corresponding vector k in the first Brillouin zone and G = q - k
    k_red_list, G_xyz_list = get_kG_list_from_q_xyz_list(q_xyz_list, recip_red_to_XYZ)

    # run phonopy to get polarization vectors and photon energies
    [ph_eigenvectors, ph_omega] = run_phonopy(phonon_file, k_red_list)

    return G_xyz_list, ph_eigenvectors, ph_omega