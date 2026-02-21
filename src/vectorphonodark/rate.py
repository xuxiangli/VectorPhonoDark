import numpy as np
import h5py

from . import constants as const
from .projection import FormFactor, BinnedMcalI


def import_all_form_factors(filename, groupname, dataname="data", verbose=True):
    """
    Utility function to import all form factors from a given HDF5 file and group.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file containing the form factors.
    groupname : str
        Name of the group within the HDF5 file where the form factors are stored.
    dataname : str, optional
        Name of the dataset within each subgroup that contains the form factors.
        Default is "data".
    verbose : bool, optional
        If True, print out the q_max values of the imported form factors. 
        Default is True.

    Returns
    -------
    form_factors : dict
        A dictionary where keys are q_max values and 
        values are the corresponding FormFactor objects.
    q_max_list : list
        A list of q_max values corresponding to the imported form factors.
    """

    form_factors = {}
    q_max_list = []

    with h5py.File(filename, "r") as h5f:
        if groupname in h5f:
            if verbose:
                print(f"    BinnedFnlm data read from {filename} in group {groupname}.")
            group = h5f[groupname]
            for key in group.keys():
                ff = FormFactor().import_hdf5(
                    filename=filename,
                    groupname=groupname + "/" + key,
                    dataname=dataname,
                    verbose=False,
                )
                q_max = ff.q_max
                form_factors[q_max] = ff
                q_max_list.append(q_max)
    
    q_max_list = np.array(sorted(q_max_list))

    if verbose:
        print(f"    Imported form factors with q_max values: {q_max_list} eV")

    return form_factors, q_max_list


class Rate:
    """
    Class to compute the binned rate using the projected VDF and Form Factor.

    The main formula is:
    R = (rho_DM / m_DM) * (v_max^2 / q_max) * sum_{l,mv,mq} G(l,mv,mq) * K(l,mv,mq)

    where G is the Wigner G coefficient and K is the projection of the integrand onto the basis.

    Parameters
    ----------
    physics_params : dict
        - fdm: function defining the dark matter form factor
        - q0_fdm: (optional) reference momentum scale for the dark matter form factor
                default: Bohr momentum
        - mass_dm: mass of the dark matter particle
        - mass_sm: mass of the Standard Model particle (e.g. nucleus)

    numerics_params : dict
        - l_max: (optional) maximum angular momentum quantum number for the projection
        - nv_max: (optional) maximum velocity wavelet index
        - nq_max: (optional) maximum momentum wavelet index

    Methods
    -------
    binned_mu_R(wG, verbose=False) : dict
        Compute the binned rate for each energy bin using the rotation matrix
        given by Wigner G coefficients.

        Return a dictionary with keys as bin indices and values as a vector of 
        the corresponding binned rates.
    """
    def __init__(self, physics_params, numerics_params, vdf, ff, verbose=False):

        if "l_max" not in numerics_params or numerics_params["l_max"] is None:
            self.l_max = min(vdf.l_max, ff.l_max)
        else:
            self.l_max = min(numerics_params["l_max"], vdf.l_max, ff.l_max)

        self.l_mod = max(vdf.l_mod, ff.l_mod)

        if "nv_max" not in numerics_params or numerics_params["nv_max"] is None:
            self.nv_max = vdf.n_max
        else:
            self.nv_max = min(numerics_params["nv_max"], vdf.n_max)

        if "nq_max" not in numerics_params or numerics_params["nq_max"] is None:
            self.nq_max = ff.n_max
        else:
            self.nq_max = min(numerics_params["nq_max"], ff.n_max)

        self.v_max = vdf.v_max
        self.q_max = ff.q_max

        self.fdm = physics_params["fdm"]
        self.q0_fdm = physics_params.get("q0_fdm", const.Q_BOHR)
        self.mass_dm = physics_params["mass_dm"]
        self.mass_sm = physics_params["mass_sm"]

        self.energy_threshold = ff.energy_threshold
        self.energy_bin_width = ff.energy_bin_width
        self.n_bins = ff.n_bins

        self.log_wavelet_q = ff.log_wavelet
        self.eps_q = ff.eps if self.log_wavelet_q else 1.0

        self.mcalKs = self._get_binned_vecK(vdf, ff, verbose=verbose)

    def binned_mu_R(self, wG, verbose=False):

        if self.l_mod != wG.lmod:
            raise ValueError("l_mod of BinnedRate and wG do not match.")
        
        G_array = np.array(wG.G_array)
        binned_muR = {}
        for idx_bin, vecK in self.mcalKs.items():
            l_max = min(self.l_max, wG.ellMax)
            lmvmq_max = self.get_lmvmq_index(l_max, l_max, l_max)

            binned_muR[idx_bin] = self.v_max**2 / self.q_max * (
                G_array[:, 0 : lmvmq_max + 1] @ vecK[0 : lmvmq_max + 1]
            )
        return binned_muR

    def get_lmvmq_index(self, l, mv, mq):
        if self.l_mod == 2 and l % 2 != 0:
            raise ValueError("l value does not satisfy l_mod=2 condition.")
        if self.l_mod == 2:
            return l*(4*l**2-6*l-1)//6 + (l+mv)*(2*l+1) + (l+mq)
        else:
            return l*(2*l-1)*(2*l+1)//3 + (l+mv)*(2*l+1) + (l+mq)
    
    def _binnedmcalI(self, verbose=False):
        physics_params = {
            "fdm": self.fdm,
            "q0_fdm": self.q0_fdm,
            "energy_threshold": self.energy_threshold,
            "energy_bin_width": self.energy_bin_width,
            "mass_dm": self.mass_dm,
            "mass_sm": self.mass_sm,
        }
        numerics_params = {
            "n_bins": self.n_bins,
            "l_max": self.l_max,
            "l_mod": self.l_mod,
            "nv_max": self.nv_max,
            "nq_max": self.nq_max,
            "v_max": self.v_max,
            "q_max": self.q_max,
            "log_wavelet_q": self.log_wavelet_q,
            "eps_q": self.eps_q,
        }

        binnedmcalI = BinnedMcalI(physics_params=physics_params, 
                                  numerics_params=numerics_params)
        binnedmcalI.project(verbose=verbose)
        return binnedmcalI

    def _get_binned_vecK(self, vdf, ff, verbose=False):

        if verbose:
            print(
                f"    Rate computation: Using {self.nv_max + 1} velocity wavelets and "
                f"{self.nq_max + 1} momentum wavelets."
            )

        # get the mcalI object
        binnedmcalI = self._binnedmcalI(verbose=verbose)

        binned_vecK = {}
        vecK_shape = (self.get_lmvmq_index(self.l_max, self.l_max, self.l_max) + 1,)
        for idx_bin in range(self.n_bins):

            # get the minimum l_max and output shape
            vecK = np.zeros(vecK_shape, dtype=float)

            for l in range(0, self.l_max + 1, self.l_mod):

                # --- Step A: Extract matrix I ---
                I_sub = binnedmcalI.mcalIs[idx_bin].mcalI[l]

                # --- Step B: Extract matrix V (VDF) ---
                rows_v = [vdf.get_lm_index(l, mv) for mv in range(-l, l + 1)]
                # shape: (2l+1, Nv)
                V_sub = vdf.f_lm_n[rows_v][:, : self.nv_max + 1]

                # --- Step C: Extract matrix F (Form Factor) ---
                rows_q = [ff.fnlms[idx_bin].get_lm_index(l, mq) for mq in range(-l, l + 1)]
                # shape: (2l+1, Nq)
                F_sub = ff.fnlms[idx_bin].f_lm_n[rows_q][:, : self.nq_max + 1]

                # --- Step D: Core computation (matrix multiplication) ---
                # Mathematical formula: K = V * I * F^T
                # Shape transformation: (2l+1, Nv) @ (Nv, Nq) @ (Nq, 2l+1)
                # -> (2l+1, 2l+1) -> vecK shape
                K_block = self.v_max**3 * V_sub @ I_sub @ F_sub.T

                # --- Step E: Fill back results into vecK ---
                # We need to calculate the corresponding flat indices in vecK
                # Assume vecK is a flattened 1D array
                target_indices = list(range(self.get_lmvmq_index(l, -l, -l), 
                                            self.get_lmvmq_index(l, l, l) + 1))
                vecK[target_indices] = K_block.flatten()

            binned_vecK[idx_bin] = vecK

        return binned_vecK
