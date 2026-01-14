import numpy as np

from . import utility


class Rate:
    def __init__(self, vdf, ff, mcalI, l_max=None, nv_list=None, nq_list=None,
                 verbose=False):

        if l_max is None:
            self.l_max = min(vdf.l_max, ff.l_max, mcalI.l_max)
        else:
            self.l_max = min(l_max, vdf.l_max, ff.l_max, mcalI.l_max)

        self.l_mod = max(vdf.l_mod, ff.l_mod, mcalI.l_mod)

        if nv_list is None:
            self.nv_list = []
        else:
            self.nv_list = nv_list

        if nq_list is None:
            self.nq_list = []
        else:
            self.nq_list = nq_list

        self.mcalK = self.get_vecK(vdf, ff, mcalI, verbose=verbose)

    def mu_R(self, wG, verbose=False):
        l_max = min(self.l_max, wG.ellMax)
        lmvmq_max = self.get_lmvmq_index(l_max, l_max, l_max)

        if self.l_mod != wG.lmod:
            raise ValueError("l_mod of Rate and wG do not match.")

        return (
            self.v_max**2 / self.q_max
            * (wG.G_array[:, 0 : lmvmq_max + 1] @ self.mcalK[0 : lmvmq_max + 1])
        )

    def get_vecK(self, vdf, ff, mcalI, verbose=False):

        if vdf.v_max != mcalI.v_max:
            raise ValueError("vdf.v_max and mcalI.v_max do not match.")
        self.v_max = vdf.v_max
        if ff.q_max != mcalI.q_max:
            raise ValueError("ff.q_max and mcalI.q_max do not match.")
        self.q_max = ff.q_max

        # get the minimum l_max and output shape
        vecK_shape = (self.get_lmvmq_index(self.l_max, self.l_max, self.l_max) + 1,)
        vecK = np.zeros(vecK_shape, dtype=float)

        # get intersection indices for nv and nq
        nv_indices = utility.get_intersection_index(
            base=self.nv_list, vdf_nv=vdf.n_list, mcalI_nv=mcalI.nv_list
        )
        nq_indices = utility.get_intersection_index(
            base=self.nq_list, ff_nq=ff.n_list, mcalI_nq=mcalI.nq_list
        )
        if nv_indices.size == 0 or nq_indices.size == 0:
            return vecK

        if verbose:
            print(
                f"      Rate computation: Using {len(nv_indices[0])} velocity wavelets and "
                f"          {len(nq_indices[0])} momentum wavelets."
            )

        idx_nv_vdf = nv_indices[0]
        idx_nv_mcal = nv_indices[1]
        idx_nq_ff = nq_indices[0]
        idx_nq_mcal = nq_indices[1]

        for l in range(self.l_max + 1):

            # --- Step A: Extract matrix I ---
            I_sub = mcalI.mcalI[l][np.ix_(idx_nv_mcal, idx_nq_mcal)]

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
            target_indices = []
            for mv in range(-l, l + 1):
                for mq in range(-l, l + 1):
                    target_indices.append(self.get_lmvmq_index(l, mv, mq))

            vecK[target_indices] = K_block.flatten()

        return vecK

    def get_lmvmq_index(self, l, mv, mq):
        return l * (2 * l - 1) * (2 * l + 1) // 3 + (l + mv) * (2 * l + 1) + (l + mq)


class BinnedRate:
    def __init__(
        self, vdf, binnedff, binnedmcalI, l_max=None, nv_list=None, nq_list=None,
        verbose=False
    ):
        if l_max is None:
            self.l_max = min(vdf.l_max, binnedff.l_max, binnedmcalI.l_max)
        else:
            self.l_max = min(l_max, vdf.l_max, binnedff.l_max, binnedmcalI.l_max)

        self.l_mod = max(vdf.l_mod, binnedff.l_mod, binnedmcalI.l_mod)

        if nv_list is None:
            self.nv_list = []
        else:
            self.nv_list = nv_list

        if nq_list is None:
            self.nq_list = []
        else:
            self.nq_list = nq_list

        self.mcalKs = self.get_binned_vecK(vdf, binnedff, binnedmcalI, verbose=verbose)

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

    def get_binned_vecK(self, vdf, binnedff, binnedmcalI, verbose=False):

        if vdf.v_max != binnedmcalI.v_max:
            raise ValueError("vdf.v_max and mcalI.v_max do not match.")
        self.v_max = vdf.v_max
        if binnedff.q_max != binnedmcalI.q_max:
            raise ValueError("ff.q_max and mcalI.q_max do not match.")
        self.q_max = binnedff.q_max

        self.n_bins = min(binnedff.n_bins, binnedmcalI.n_bins)

        binned_vecK = {}
        for idx_bin in range(self.n_bins):
            vecK = Rate(
                vdf,
                binnedff.fnlms[idx_bin],
                binnedmcalI.mcalIs[idx_bin],
                l_max=self.l_max,
                nv_list=self.nv_list,
                nq_list=self.nq_list,
                verbose=verbose,
            ).mcalK
            binned_vecK[idx_bin] = vecK
        return binned_vecK

    def get_lmvmq_index(self, l, mv, mq):
        return l * (2 * l - 1) * (2 * l + 1) // 3 + (l + mv) * (2 * l + 1) + (l + mq)
