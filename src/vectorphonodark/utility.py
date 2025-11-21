import numpy as np
import math
import numba
import os
import csv
import quaternionic
from importlib import util

from . import constants as const
from . import phonopy_funcs


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


def writeFnlm_csv(csvsave_name, f_nlm_coeffs={}, info={}, use_gvar=False) -> None:
    """
    Write function coefficients to a CSV file.

    Args:
        csvsave_name: 
            str: The path to the CSV file where coefficients will be saved.
        f_nlm_coeffs: 
            dict: A dictionary where keys are (n, l, m) tuples and values are function coefficients.
        basis: 
            dict: A dictionary of basis parameters to include in the CSV header.
        use_gvar: 
            bool: Whether the function coefficients are gvar objects (default is False).
    """

    makeHeader = not os.path.exists(csvsave_name)
    with open(csvsave_name, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quoting=csv.QUOTE_MINIMAL)
        if makeHeader:
            bparams = [r'#'] + [str(lbl) + ': ' + str(prm)
                                    for lbl,prm in info.items()]
            writer.writerow(bparams)
            header = [r'#', 'n', 'l', 'm', 'f.mean', 'f.sdev']
            writer.writerow(header)
        for nlm in f_nlm_coeffs.keys():
            f = f_nlm_coeffs[nlm]
            if use_gvar:
                mean, std = f.mean, f.sdev
            else:
                mean, std = f, 0
            newline = [nlm[0], nlm[1], nlm[2], mean, std]
            writer.writerow(newline)


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


def getQ(theta, phi):
    axisphi = phi + np.pi/2 #stationary under R
    axR = theta/2 
    qr = np.cos(axR)
    qi = np.sin(axR) * np.cos(axisphi)
    qj = np.sin(axR) * np.sin(axisphi)
    qk = 0. 
    return quaternionic.array(qr, qi, qj, qk)


def run_phonopy(phonon_file, k_red) -> list[np.ndarray]:
    """
    Run phonopy to compute phonon eigenvectors and frequencies at given k-points.

    Args:
        phonon_file: 
            Phonopy object: The phonopy object containing the phonon data.
        k_red: 
            np.ndarray: An array of shape (N, 3) representing the reduced k-points.

    Returns:
        list[np.ndarray]: A list containing two elements:
            - eigenvectors: An array of shape (num_modes, num_atoms, 3) representing the phonon eigenvectors.
            - omega: An array of shape (num_modes,) representing the phonon frequencies in eV.
    """

    # run phonopy in mesh mode 
    phonon_file.run_qpoints(k_red, with_eigenvectors=True)

    mesh_dict = phonon_file.get_qpoints_dict()

    eigenvectors_pre = mesh_dict['eigenvectors']

    # convert frequencies to correct units
    omega = 2*const.PI*(const.THz_To_eV)*mesh_dict['frequencies'][0]

    num_atoms = phonon_file.primitive.get_number_of_atoms()
    num_modes = 3*num_atoms 

    # q, nu, i, alpha
    eigenvectors = np.zeros((num_modes, num_atoms, 3), dtype=complex)

    # sort the eigenvectors
    for nu in range(num_modes):
        eigenvectors[nu][:][:] = np.array(np.array_split(eigenvectors_pre.T[nu], num_atoms)).reshape(num_atoms, 3)

    return [eigenvectors, omega]


@numba.njit
def get_kG_from_q_red(q_red_vec, q_red_to_XYZ):
    """
        q_red_vec: q vector in reduced coordinates
        q_red_to_XYZ: matrix converting q in reduced coordinates to XYZ

        output: [k_red_vec, G_red_vec]: the k and G vectors in reduced coordinates 

    """
    set_of_closest_G_red = np.zeros((8, 3), dtype=np.float64)

    set_of_closest_G_red[0] = [math.floor(q_red_vec[0]), math.floor(q_red_vec[1]), math.floor(q_red_vec[2])]
    set_of_closest_G_red[1] = [math.floor(q_red_vec[0]), math.floor(q_red_vec[1]), math.ceil(q_red_vec[2])]
    set_of_closest_G_red[2] = [math.floor(q_red_vec[0]), math.ceil(q_red_vec[1]), math.floor(q_red_vec[2])]
    set_of_closest_G_red[3] = [math.ceil(q_red_vec[0]), math.floor(q_red_vec[1]), math.floor(q_red_vec[2])]
    set_of_closest_G_red[4] = [math.floor(q_red_vec[0]), math.ceil(q_red_vec[1]), math.ceil(q_red_vec[2])]
    set_of_closest_G_red[5] = [math.ceil(q_red_vec[0]), math.floor(q_red_vec[1]), math.ceil(q_red_vec[2])]
    set_of_closest_G_red[6] = [math.ceil(q_red_vec[0]), math.ceil(q_red_vec[1]), math.floor(q_red_vec[2])]
    set_of_closest_G_red[7] = [math.ceil(q_red_vec[0]), math.ceil(q_red_vec[1]), math.ceil(q_red_vec[2])]

    # q_XYZ_vec = q_red_to_XYZ @ q_red_vec
    q_XYZ_vec = [q_red_to_XYZ[0, 0]*q_red_vec[0] + q_red_to_XYZ[0, 1]*q_red_vec[1] + q_red_to_XYZ[0, 2]*q_red_vec[2],
                  q_red_to_XYZ[1, 0]*q_red_vec[0] + q_red_to_XYZ[1, 1]*q_red_vec[1] + q_red_to_XYZ[1, 2]*q_red_vec[2],
                  q_red_to_XYZ[2, 0]*q_red_vec[0] + q_red_to_XYZ[2, 1]*q_red_vec[1] + q_red_to_XYZ[2, 2]*q_red_vec[2]]

    first = True

    for vec in set_of_closest_G_red:

        # diff_vec = q_XYZ_vec - q_red_to_XYZ @ vec
        diff_vec = [q_XYZ_vec[0] - (q_red_to_XYZ[0, 0]*vec[0] + q_red_to_XYZ[0, 1]*vec[1] + q_red_to_XYZ[0, 2]*vec[2]),
                    q_XYZ_vec[1] - (q_red_to_XYZ[1, 0]*vec[0] + q_red_to_XYZ[1, 1]*vec[1] + q_red_to_XYZ[1, 2]*vec[2]),
                    q_XYZ_vec[2] - (q_red_to_XYZ[2, 0]*vec[0] + q_red_to_XYZ[2, 1]*vec[1] + q_red_to_XYZ[2, 2]*vec[2])]
        diff_vec_sq = diff_vec[0]*diff_vec[0] + diff_vec[1]*diff_vec[1] + diff_vec[2]*diff_vec[2]

        # if first:
        #     min_dist_sq = np.dot(diff_vec, diff_vec)
        #     first = False

        # if np.dot(diff_vec, diff_vec) <= min_dist_sq:
        #     min_vec = vec

        if first:
            min_dist_sq = diff_vec_sq
            first = False

        if diff_vec_sq <= min_dist_sq:
            min_vec = vec

    G_red_vec = min_vec

    # k_red_vec = np.array(q_red_vec) - np.array(G_red_vec)
    k_red_vec = [q_red_vec[0] - G_red_vec[0],
                 q_red_vec[1] - G_red_vec[1],
                 q_red_vec[2] - G_red_vec[2]]

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

    set_of_closest_G_red[0] = np.array([math.floor(q_red_vec[0]), math.floor(q_red_vec[1]), math.floor(q_red_vec[2])])
    set_of_closest_G_red[1] = np.array([math.floor(q_red_vec[0]), math.floor(q_red_vec[1]), math.ceil(q_red_vec[2])])
    set_of_closest_G_red[2] = np.array([math.floor(q_red_vec[0]), math.ceil(q_red_vec[1]), math.floor(q_red_vec[2])])
    set_of_closest_G_red[3] = np.array([math.ceil(q_red_vec[0]), math.floor(q_red_vec[1]), math.floor(q_red_vec[2])])
    set_of_closest_G_red[4] = np.array([math.floor(q_red_vec[0]), math.ceil(q_red_vec[1]), math.ceil(q_red_vec[2])])
    set_of_closest_G_red[5] = np.array([math.ceil(q_red_vec[0]), math.floor(q_red_vec[1]), math.ceil(q_red_vec[2])])
    set_of_closest_G_red[6] = np.array([math.ceil(q_red_vec[0]), math.ceil(q_red_vec[1]), math.floor(q_red_vec[2])])
    set_of_closest_G_red[7] = np.array([math.ceil(q_red_vec[0]), math.ceil(q_red_vec[1]), math.ceil(q_red_vec[2])])

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
def get_kG_list_from_q_xyz_list(q_xyz_list, recip_red_to_XYZ) -> tuple[np.ndarray, np.ndarray]:
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


def get_G_eigenvectors_omega_from_q_xyz(q_xyz_list, phonon_file, phonopy_params) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Get q_xyz, G_xyz, eigenvectors, and omega from a q_sph mesh.
    """

    # reciprocal matrix to convert q in reduced coordinates to XYZ
    recip_red_to_XYZ = np.array(phonopy_params['recip_red_to_XYZ'])

    # get corresponding vector k in the first Brillouin zone and G = q - k
    k_red_list, G_xyz_list = get_kG_list_from_q_xyz_list(q_xyz_list, recip_red_to_XYZ)

    # run phonopy to get polarization vectors and photon energies
    [ph_eigenvectors, ph_omega] = phonopy_funcs.run_phonopy(phonon_file, k_red_list)

    return G_xyz_list, ph_eigenvectors, ph_omega