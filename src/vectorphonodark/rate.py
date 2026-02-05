import numpy as np

from . import constants as const
from . import utility
from .projection import McalI, BinnedMcalI


class Rate:
    def __init__(self, physics_params, numerics_params, vdf, ff):

        if "l_max" in numerics_params:
            self.l_max = min(numerics_params["l_max"], vdf.l_max, ff.l_max)
        else:
            self.l_max = min(vdf.l_max, ff.l_max)

        self.l_mod = max(vdf.l_mod, ff.l_mod)

        if "nv_list" in numerics_params:
            self.nv_list = sorted(list(
                set(numerics_params["nv_list"]) & set(vdf.n_list)
            ))
        else:
            self.nv_list = vdf.n_list

        if "nq_list" in numerics_params:
            self.nq_list = sorted(list(
                set(numerics_params["nq_list"]) & set(ff.n_list)
            ))
        else:
            self.nq_list = ff.n_list

        self.v_max = vdf.v_max
        self.q_max = ff.q_max

        self.fdm = physics_params["fdm"]
        self.q0_fdm = physics_params.get("q0_fdm", const.Q_BOHR)
        self.energy = physics_params["energy"]
        self.mass_dm = physics_params["mass_dm"]
        self.mass_sm = physics_params["mass_sm"]

        self.log_wavelet_q = ff.log_wavelet
        self.eps_q = ff.eps if self.log_wavelet_q else 1.0

        self.mcalK = self.get_vecK(vdf, ff)
    
    def calc_mcalI(self):
        physics_params = {
            "fdm": self.fdm,
            "q0_fdm": self.q0_fdm,
            "energy": self.energy,
            "mass_dm": self.mass_dm,
            "mass_sm": self.mass_sm,
        }
        numerics_params = {
            "l_max": self.l_max,
            "l_mod": self.l_mod,
            "nv_list": self.nv_list,
            "nq_list": self.nq_list,
            "v_max": self.v_max,
            "q_max": self.q_max,
            "log_wavelet_q": self.log_wavelet_q,
            "eps_q": self.eps_q,
        }

        mcalI = McalI(physics_params=physics_params, numerics_params=numerics_params)
        mcalI.project(verbose=False)
        return mcalI
        

    def mu_R(self, wG):
        l_max = min(self.l_max, wG.ellMax)
        lmvmq_max = self.get_lmvmq_index(l_max, l_max, l_max)

        if self.l_mod != wG.lmod:
            raise ValueError("l_mod of Rate and wG do not match.")

        return (
            self.v_max**2 / self.q_max
            * (wG.G_array[:, 0 : lmvmq_max + 1] @ self.mcalK[0 : lmvmq_max + 1])
        )

    def get_vecK(self, vdf, ff, verbose=False):

        # get the minimum l_max and output shape
        vecK_shape = (self.get_lmvmq_index(self.l_max, self.l_max, self.l_max) + 1,)
        vecK = np.zeros(vecK_shape, dtype=float)

        # get intersection indices for nv and nq
        nv_indices = utility.get_intersection_index(
            base=self.nv_list, vdf_nv=vdf.n_list
        )
        nq_indices = utility.get_intersection_index(
            base=self.nq_list, ff_nq=ff.n_list
        )
        if nv_indices.size == 0 or nq_indices.size == 0:
            return vecK

        if verbose:
            print(
                f"    Rate computation: Using {len(nv_indices[0])} velocity wavelets and "
                f"{len(nq_indices[0])} momentum wavelets."
            )

        # get the mcalI object
        mcalI = self.calc_mcalI()

        idx_nv_vdf = nv_indices[0]
        idx_nq_ff = nq_indices[0]

        for l in range(self.l_max + 1):

            # --- Step A: Extract matrix I ---
            # I_sub = mcalI.mcalI[l][np.ix_(idx_nv_mcal, idx_nq_mcal)]
            I_sub = mcalI.mcalI[l]

            # --- Step B: Extract matrix V (VDF) ---
            rows_v = [vdf.get_lm_index(l, mv) for mv in range(-l, l + 1)]
            # shape: (2l+1, Nv)
            V_sub = vdf.f_lm_n[rows_v][:, idx_nv_vdf]

            # --- Step C: Extract matrix F (Form Factor) ---
            rows_q = [ff.get_lm_index(l, mq) for mq in range(-l, l + 1)]
            # shape: (2l+1, Nq)
            F_sub = ff.f_lm_n[rows_q][:, idx_nq_ff]

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

        return vecK

    def get_lmvmq_index(self, l, mv, mq):
        return l * (2 * l - 1) * (2 * l + 1) // 3 + (l + mv) * (2 * l + 1) + (l + mq)


class BinnedRate:
    def __init__(self, physics_params, numerics_params, vdf, ff):

        if "l_max" in numerics_params:
            self.l_max = min(numerics_params["l_max"], vdf.l_max, ff.l_max)
        else:
            self.l_max = min(vdf.l_max, ff.l_max)

        self.l_mod = max(vdf.l_mod, ff.l_mod)

        if "nv_list" in numerics_params:
            self.nv_list = sorted(list(
                set(numerics_params["nv_list"]) & set(vdf.n_list)
            ))
        else:
            self.nv_list = vdf.n_list

        if "nq_list" in numerics_params:
            self.nq_list = sorted(list(
                set(numerics_params["nq_list"]) & set(ff.n_list)
            ))
        else:
            self.nq_list = ff.n_list

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

        self.mcalKs = self.get_binned_vecK(vdf, ff)
    
    def calc_binnedmcalI(self, verbose=True):
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
            "nv_list": self.nv_list,
            "nq_list": self.nq_list,
            "v_max": self.v_max,
            "q_max": self.q_max,
            "log_wavelet_q": self.log_wavelet_q,
            "eps_q": self.eps_q,
        }

        binnedmcalI = BinnedMcalI(physics_params=physics_params, 
                                  numerics_params=numerics_params)
        binnedmcalI.project(verbose=verbose)
        return binnedmcalI

    def binned_mu_R(self, wG, verbose=False):
        G_array = np.array(wG.G_array)
        binned_muR = {}
        for idx_bin, vecK in self.mcalKs.items():
            l_max = min(self.l_max, wG.ellMax)
            lmvmq_max = self.get_lmvmq_index(l_max, l_max, l_max)

            if self.l_mod != wG.lmod:
                raise ValueError("l_mod of BinnedRate and wG do not match.")

            binned_muR[idx_bin] = (
                self.v_max**2
                / self.q_max
                * (G_array[:, 0 : lmvmq_max + 1] @ vecK[0 : lmvmq_max + 1])
            )
        return binned_muR

    def get_binned_vecK(self, vdf, ff, verbose=False):

        # get intersection indices for nv and nq
        nv_indices = utility.get_intersection_index(
            base=self.nv_list, vdf_nv=vdf.n_list
        )
        nq_indices = utility.get_intersection_index(
            base=self.nq_list, ff_nq=ff.n_list
        )
        if nv_indices.size == 0 or nq_indices.size == 0:
            return vecK

        if verbose:
            print(
                f"    Rate computation: Using {len(nv_indices[0])} velocity wavelets and "
                f"{len(nq_indices[0])} momentum wavelets."
            )

        # get the mcalI object
        binnedmcalI = self.calc_binnedmcalI()

        idx_nv_vdf = nv_indices[0]
        idx_nq_ff = nq_indices[0]

        binned_vecK = {}
        vecK_shape = (self.get_lmvmq_index(self.l_max, self.l_max, self.l_max) + 1,)
        for idx_bin in range(self.n_bins):

            # get the minimum l_max and output shape
            vecK = np.zeros(vecK_shape, dtype=float)

            for l in range(self.l_max + 1):

                # --- Step A: Extract matrix I ---
                I_sub = binnedmcalI.mcalIs[idx_bin].mcalI[l]

                # --- Step B: Extract matrix V (VDF) ---
                rows_v = [vdf.get_lm_index(l, mv) for mv in range(-l, l + 1)]
                # shape: (2l+1, Nv)
                V_sub = vdf.f_lm_n[rows_v][:, idx_nv_vdf]

                # --- Step C: Extract matrix F (Form Factor) ---
                rows_q = [ff.fnlms[idx_bin].get_lm_index(l, mq) for mq in range(-l, l + 1)]
                # shape: (2l+1, Nq)
                F_sub = ff.fnlms[idx_bin].f_lm_n[rows_q][:, idx_nq_ff]

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

    def get_lmvmq_index(self, l, mv, mq):
        return l * (2 * l - 1) * (2 * l + 1) // 3 + (l + mv) * (2 * l + 1) + (l + mq)
