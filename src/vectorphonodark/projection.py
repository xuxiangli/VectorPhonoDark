import numpy as np
import time
import os
import csv
import h5py

from . import constants as const
from . import utility
from .utility import C_GREEN, C_CYAN, C_RESET
from . import physics
# from . import analytic
try:
    from . import analytic_cy as analytic
except ImportError:
    print("Warning: Could not load Cython module, falling back to Python version.")
    from . import analytic


class Fnlm:
    def __init__(self, l_max=-1, l_mod=1, n_max=-1, info=None):
        if l_mod not in [1, 2]:
            raise ValueError("l_mod must be either 1 (all l) or 2 (even l only).")
        
        self.l_max = l_max
        self.l_mod = l_mod
        self.n_max = n_max
        self.info = {} if info is None else info
        
        self.f_lm_n = np.array([])

    def get_lm_index(self, l, m):
        if abs(m) > l or l > self.l_max or l % self.l_mod != 0:
            raise ValueError("Invalid (l, m) values.")
        if self.l_mod == 2:
            return l*(l-1)//2 + l + m
        else:
            return l**2 + l + m

    def get_n_index(self, n):
        if n > self.n_max:
            raise ValueError("n value exceeds n_max.")
        return n

    def export_csv(self, filename, write_info=True, verbose=True):
        if not filename.endswith(".csv"):
            filename += ".csv"
        makeHeader = not os.path.exists(filename)
        with open(filename, mode="a") as file:
            writer = csv.writer(file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
            if makeHeader:
                if write_info:
                    bparams = [r"#"] + [
                        str(key) + ":" + str(value) 
                        for key, value in self.info.items()
                    ]
                    writer.writerow(bparams)
                header = [r"#n", "l", "m", "f_lm_n"]
                writer.writerow(header)
            for l in range(0, self.l_max + 1, self.l_mod):
                for m in range(-l, l + 1):
                    idx_lm = self.get_lm_index(l, m)
                    for n in range(self.n_max + 1):
                        row = [n, l, m, self.f_lm_n[idx_lm, n]]
                        writer.writerow(row)

        if verbose:
            print(f"    Fnlm data written to {C_GREEN}{filename}{C_RESET}.")

    def import_csv(self, filename, verbose=True):
        """
        Import fnlm data from a CSV file.
        Will reset existing data.
        """
        # reset data
        self.info = {}
        self.n_index_map = {}

        if not filename.endswith(".csv"):
            filename += ".csv"
        with open(filename, mode="r") as file:
            reader = csv.reader(file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
            data = {}
            n_max = -1
            l_list = set()
            for row in reader:
                if row[0].startswith("#"):
                    if len(row) > 1 and ":" in row[1]:
                        for item in row[1:]:
                            key, value = item.split(":", 1)
                            self.info[key] = value
                    continue
                else:
                    n, l, m = int(row[0]), int(row[1]), int(row[2])
                    value = float(row[3])
                    n_max = max(n_max, n)
                    l_list.add(l)
                    data[(n, l, m)] = value

            self.l_max = max(l_list)
            self.l_mod = 2 if all(l % 2 == 0 for l in l_list) else 1
            self.n_max = n_max

            self.f_lm_n = np.zeros(
                (self.get_lm_index(self.l_max, self.l_max) + 1, self.n_max + 1),
                dtype=float,
            )
            for (n, l, m), value in data.items():
                idx_lm = self.get_lm_index(l, m)
                idx_n = self.get_n_index(n)
                self.f_lm_n[idx_lm, idx_n] = value

        if verbose:
            print(f"    Fnlm data read from {C_GREEN}{filename}{C_RESET}.")

    def export_hdf5(
        self, filename, groupname, dataname="data", write_info=True, verbose=True
    ):
        if not filename.endswith(".hdf5"):
            filename += ".hdf5"
        with h5py.File(filename, "a") as h5f:
            grp = h5f.require_group(groupname)
            if dataname in grp:
                del grp[dataname]
            dset = grp.create_dataset(dataname, data=self.f_lm_n)
            dset.attrs["l_max"] = self.l_max
            dset.attrs["l_mod"] = self.l_mod
            dset.attrs["n_max"] = self.n_max
            if write_info:
                for key, value in self.info.items():
                    dset.attrs[key] = value
        if verbose:
            print(
                f"    Fnlm data written to {C_GREEN}{filename}{C_RESET}",
                f"in group {C_CYAN}{groupname}/{dataname}{C_RESET}.",
            )

    def import_hdf5(self, filename, groupname, dataname="data", verbose=True):
        """
        Import fnlm data from an HDF5 file.
        Will reset existing data.
        """
        if not filename.endswith(".hdf5"):
            filename += ".hdf5"
        with h5py.File(filename, "r") as h5f:
            grp = h5f[groupname]
            dset = grp[dataname]
            self.f_lm_n = dset[()]
            self.l_max = dset.attrs["l_max"]
            self.l_mod = dset.attrs["l_mod"]
            self.n_max = dset.attrs["n_max"]
            self.info = {
                key: dset.attrs[key]
                for key in dset.attrs
                if key not in ["l_max", "l_mod", "n_max"]
            }

        if verbose:
            print(
                f"    Fnlm data read from {C_GREEN}{filename}{C_RESET}",
                f"in group {C_CYAN}{groupname}/{dataname}{C_RESET}.",
            )


class BinnedFnlm:
    def __init__(self, n_bins=0, l_max=-1, l_mod=1, n_max=-1, info=None):
        if l_mod not in [1, 2]:
            raise ValueError("l_mod must be either 1 (all l) or 2 (even l only).")
        
        self.n_bins = n_bins
        self.l_max = l_max
        self.l_mod = l_mod
        self.n_max = n_max
        self.info = {} if info is None else info

        self.fnlms = {}

    def _check_consistency(self):
        l_max_list = [fnlm.l_max for fnlm in self.fnlms.values()]
        if not all(l_max == self.l_max for l_max in l_max_list):
            raise ValueError("Inconsistent l_max among bins.")
        
        l_mod_list = [fnlm.l_mod for fnlm in self.fnlms.values()]
        if not all(l_mod == self.l_mod for l_mod in l_mod_list):
            raise ValueError("Inconsistent l_mod among bins.")
        
        n_max_list = [fnlm.n_max for fnlm in self.fnlms.values()]
        if not all(n_max == self.n_max for n_max in n_max_list):
            raise ValueError("Inconsistent n_max among bins.")
        
    def export_csv(self, filename, write_info=True, verbose=True):
        if filename.endswith(".csv"):
            filename = filename[:-4]
        for idx_bin in range(self.n_bins):
            bin_filename = filename + "_bin_" + str(idx_bin) + ".csv"
            self.fnlms[idx_bin].export_csv(
                bin_filename, write_info=write_info, verbose=False
            )
        if verbose:
            print(
                f"    BinnedFnlm data written to {C_GREEN}{filename}_bin_*.csv{C_RESET} files."
            )

    def import_csv(self, filename, verbose=True):
        if filename.endswith(".csv"):
            filename = filename[:-4]

        self.info = {}
        self.fnlms = {}
        idx_bin = 0
        while True:
            try:
                bin_filename = filename + "_bin_" + str(idx_bin) + ".csv"
                fnlm = Fnlm()
                fnlm.import_csv(bin_filename, verbose=False)
                self.fnlms[idx_bin] = fnlm
                self.info |= fnlm.info
                idx_bin += 1
            except FileNotFoundError:
                break
        self.n_bins = idx_bin

        if self.n_bins == 0:
            raise ValueError("No files found for BinnedFnlm import.")

        self.l_max = self.fnlms[0].l_max
        self.l_mod = self.fnlms[0].l_mod
        self.n_max = self.fnlms[0].n_max
        self._check_consistency()

        if verbose:
            print(
                f"    BinnedFnlm data read from {C_GREEN}{filename}_bin_*.csv{C_RESET} files."
            )

    def export_hdf5(
        self, filename, groupname, dataname="data", write_info=True, verbose=True
    ):
        if not filename.endswith(".hdf5"):
            filename += ".hdf5"
        with h5py.File(filename, "a") as h5f:
            grp = h5f.require_group(groupname)
            grp.attrs["n_bins"] = self.n_bins
            grp.attrs["l_max"] = self.l_max
            grp.attrs["l_mod"] = self.l_mod
            grp.attrs["n_max"] = self.n_max
            for key, value in self.info.items():
                grp.attrs[key] = value
            for idx_bin, fnlm in self.fnlms.items():
                grp.require_group(f"bin_{idx_bin}")
                fnlm.export_hdf5(
                    filename,
                    f"{groupname}/bin_{idx_bin}",
                    dataname,
                    write_info=write_info,
                    verbose=False,
                )

        if verbose:
            print(
                f"    BinnedFnlm data written to {C_GREEN}{filename}{C_RESET}",
                f"in group {C_CYAN}{groupname}/bin_*{C_RESET}.",
            )

    def import_hdf5(self, filename, groupname, dataname="data", verbose=True):
        if not filename.endswith(".hdf5"):
            filename += ".hdf5"
        with h5py.File(filename, "r") as h5f:
            grp = h5f[groupname]
            self.n_bins = grp.attrs["n_bins"]
            self.l_max = grp.attrs["l_max"]
            self.l_mod = grp.attrs["l_mod"]
            self.n_max = grp.attrs["n_max"]
            self.info = {
                key: grp.attrs[key]
                for key in grp.attrs
                if key not in ["n_bins", "l_max", "l_mod", "n_max"]
            }
            self.fnlms = {}
            for idx_bin in range(self.n_bins):
                grp[f"bin_{idx_bin}"]
                fnlm = Fnlm()
                fnlm.import_hdf5(
                    filename, f"{groupname}/bin_{idx_bin}", dataname, verbose=False
                )
                self.fnlms[idx_bin] = fnlm
        
        self._check_consistency()

        if verbose:
            print(
                f"    BinnedFnlm data read from {C_GREEN}{filename}{C_RESET}",
                f"in group {C_CYAN}{groupname}bin_*{C_RESET}.",
            )


class VDF(Fnlm):
    """
    Velocity Distribution Function class.

    Parameters
    ----------
    physics_params : dict
        Physics parameters required for VDF projection.
            - vdf: the velocity distribution function.
            - vdf_params: parameters for the velocity distribution function.
            - model: (optional) name of the VDF model.

    numerics_params : dict
        Numerical parameters required for VDF projection.
            - v_max: reference velocity scale.
            - l_max: maximum angular momentum quantum number.
            - l_mod: (optional) denotes parity of l values (1 for all, 2 for even only).
            - n_list: list of radial quantum numbers.
    """

    def __init__(self, physics_params, numerics_params):
        l_max = numerics_params.get("l_max", -1)
        l_mod = numerics_params.get("l_mod", 1)
        n_max = numerics_params.get("n_max", -1)
        super().__init__(l_max=l_max, l_mod=l_mod, n_max=n_max)

        self.vdf = physics_params.get("vdf", None)
        self.vdf_params = physics_params.get("vdf_params", {})

        self.v_max = numerics_params.get("v_max", 1.0)

        self.f_lm_n = np.array([])
        for key, value in self.vdf_params.items():
            self.info["vdf_param_" + key] = value
        self.info["v_max"] = self.v_max

        if "model" in physics_params:
            self.info["model"] = physics_params["model"]

    def import_csv(self, filename, verbose=True):
        super().import_csv(filename, verbose=verbose)
        keys_to_load = [
            # "l_max",
            # "l_mod",
            # "n_max",
            "v_max"
        ]
        for key in keys_to_load:
            if key in self.info:
                setattr(self, key, self.info[key])
            else:
                raise KeyError(f"Key '{key}' not found in CSV info.")

    def import_hdf5(self, filename, groupname, dataname="data", verbose=True):
        super().import_hdf5(filename, groupname, dataname, verbose)
        keys_to_load = [
            # "l_max",
            # "l_mod",
            # "n_max",
            "v_max"
        ]
        for key in keys_to_load:
            if key in self.info:
                setattr(self, key, self.info[key])
            else:
                raise KeyError(f"Key '{key}' not found in HDF5 info.")

    def project(self, params, verbose=True):

        if verbose:
            print(
                "\n    Starting projection of velocity distribution function"
                " onto basis functions..."
            )
            start_total_time = time.time()

        l_max = self.l_max
        l_mod = self.l_mod
        n_max = self.n_max

        vdf = self.vdf
        vdf_params = self.vdf_params
        v_max = self.v_max

        n_a, n_b, n_c = params["n_grid"]
        # p_a, p_b, p_c = params.get("power_grid", (1, 1, 1))

        if verbose:
            if "model" in self.info:
                print(f"    Using VDF model: {self.info['model']}")
            print(f"    Parameters for VDF: {vdf_params}")
            print(f"    Reference velocity v_max = {v_max:.2e}.")
            print(f"    Projecting onto basis with l_max={l_max}, l_mod={l_mod}, n_max={n_max}.")
            print(f"    Grid size: n_a={n_a}, n_b={n_b}, n_c={n_c}")
            # print(f"    Grid rescaled: power_a={p_a}, power_b={p_b}, power_c={p_c}")
            print("\n    Generate grids and calculating form factor...")
            start_time = time.time()

        # Prepare grid points and function values
        lm_list = [(l, m) for l in range(0, l_max + 1, l_mod) for m in range(-l, l + 1)]
        v_xyz_list, y_lm_vals, jacob_vals = utility.gen_mesh_ylm_jacob(
            lm_list, v_max, n_a, n_b, n_c  # , p_a, p_b, p_c
        )

        vdf_vals = np.array(
            [vdf(v_vec, **vdf_params) for v_vec in v_xyz_list]
        ).reshape(n_a, n_b, n_c)
        del v_xyz_list

        if verbose:
            end_time = time.time()
            print(f"    VDF calculation completed in "
                  f"{end_time - start_time:.2f} seconds.")
            print("\n    Projecting VDF onto basis functions...")

        # Project vdf onto basis functions
        f_nlm = utility.proj_get_f_nlm(
            n_max=n_max, lm_list=lm_list,
            func_vals=vdf_vals, y_lm_vals=y_lm_vals, jacob_vals=jacob_vals,
            n_a=n_a, # p_a,
        )
        del vdf_vals, y_lm_vals, jacob_vals

        # Store results
        self.f_lm_n = np.zeros(
            (self.get_lm_index(l_max, l_max) + 1, n_max + 1), dtype=float
        )
        for l in range(0, l_max + 1, l_mod):
            for m in range(-l, l + 1):
                idx_lm = self.get_lm_index(l, m)
                for n in range(n_max + 1):
                    self.f_lm_n[idx_lm, n] = f_nlm.get((n, l, m), 0.0)

        if verbose:
            print("    Projection completed.")
            end_total_time = time.time()
            print(f"    Total projection time: "
                  f"{end_total_time - start_total_time:.2f} seconds.")


class FormFactor(BinnedFnlm):
    """
    Form Factor class.

    Parameters
    ----------
    physics_params : dict
        Physics parameters required for Form Factor projection.
            - energy_threshold: minimum energy transfer.
            - energy_bin_width: width of each energy bin.
            - energy_max_factor: (optional) factor to determine maximum energy.
            - model: (optional) name of the Form Factor model.

    numerics_params : dict
        Numerical parameters required for Form Factor projection.
            - q_max: reference momentum scale.
            - l_max: maximum angular momentum quantum number.
            - n_list: list of radial quantum numbers.
            - q_cut: (optional) whether to compute q_cut from Debye-Waller factor.
            - log_wavelet: (optional) whether to use log wavelet mesh.
    """

    def __init__(self, physics_params, numerics_params):
        l_max = numerics_params.get("l_max", -1)
        l_mod = numerics_params.get("l_mod", 1)
        n_max = numerics_params.get("n_max", -1)
        super().__init__(n_bins=0, l_max=l_max, l_mod=l_mod, n_max=n_max)

        self.energy_threshold = physics_params.get("energy_threshold", 0.0)
        self.energy_bin_width = physics_params.get("energy_bin_width", 0.0)
        self.energy_max_factor = physics_params.get("energy_max_factor", 1.2)

        self.q_max = numerics_params.get("q_max", 1.0)
        self.q_cut = numerics_params.get("q_cut", False)
        self.log_wavelet = numerics_params.get("log_wavelet", False)

        # Store additional info
        self.info["energy_threshold"] = self.energy_threshold
        self.info["energy_bin_width"] = self.energy_bin_width
        self.info["energy_max_factor"] = self.energy_max_factor
        self.info["q_cut"] = self.q_cut
        self.info["log_wavelet"] = self.log_wavelet

        if "model" in physics_params:
            self.info["model"] = physics_params["model"]

    def import_csv(self, filename, verbose=True):
        super().import_csv(filename, verbose=verbose)
        keys_to_load = [
            # "n_bins",
            # "l_max",
            # "l_mod",
            # "n_max",
            "q_max",
            "energy_threshold",
            "energy_bin_width",
            # "energy_max_factor",
            "q_cut",
            "log_wavelet",
        ]
        for key in keys_to_load:
            if key in self.info:
                setattr(self, key, self.info[key])
            else:
                raise KeyError(f"Key '{key}' not found in CSV info.")
        if self.log_wavelet:
            if "log_wavelet_eps" in self.info:
                self.eps = self.info["log_wavelet_eps"]
            else:
                raise KeyError("Key 'log_wavelet_eps' not found in CSV info.")

    def import_hdf5(self, filename, groupname, dataname="data", verbose=True):
        super().import_hdf5(filename, groupname, dataname, verbose=verbose)
        keys_to_load = [
            # "n_bins",
            # "l_max",
            # "l_mod",
            # "n_max",
            "q_max",
            "energy_threshold",
            "energy_bin_width",
            # "energy_max_factor",
            "q_cut",
            "log_wavelet",
        ]
        for key in keys_to_load:
            if key in self.info:
                setattr(self, key, self.info[key])
            else:
                raise KeyError(f"Key '{key}' not found in HDF5 info.")
        if self.log_wavelet:
            if "log_wavelet_eps" in self.info:
                self.eps = self.info["log_wavelet_eps"]
            else:
                raise KeyError("Key 'log_wavelet_eps' not found in HDF5 info.")
        
    def project(self, params, verbose=False):

        if verbose:
            print("\n    Starting projection of form factor onto basis functions...")
            start_total_time = time.time()

        # Phonon data
        phonon_file, phonopy_params, c_dict, n_DW_params = physics.get_material_data(
            params["material_input"], params["physics_model_input"], params["numerics_input"]
        )

        l_max = self.l_max
        l_mod = self.l_mod
        n_max = self.n_max

        energy_threshold = self.energy_threshold
        energy_bin_width = self.energy_bin_width
        energy_max_factor = self.energy_max_factor
        energy_max = physics.get_energy_max(phonon_file, factor=energy_max_factor)
        energy_bin_num = int((energy_max-energy_threshold)/energy_bin_width)+1
        self.n_bins = energy_bin_num
        self.info["n_bins"] = self.n_bins

        q_max = self.q_max
        q_cut_option = self.q_cut
        q_max = physics.get_q_max(
            q_max=q_max,
            q_cut_option=q_cut_option,
            phonon_file=phonon_file,
            atom_masses=phonopy_params["atom_masses"],
            verbose=verbose,
        )
        self.q_max = q_max
        self.info["q_max"] = self.q_max
        log_wavelet = self.log_wavelet

        n_a, n_b, n_c = params["n_grid"]
        # p_a, p_b, p_c = params.get("power_grid", (1, 1, 1))

        if verbose:
            if "model" in self.info:
                print(f"    Model: {self.info['model']}")
            if log_wavelet:
                print("    Using log wavelet mesh for projection.")
            else:
                print(f"    Standard (power) wavelet basis in q.")
            print(f"    Reference momentum q_max = {q_max:.2e} eV.")
            print(f"    Projecting onto basis with l_max={l_max}, l_mod={l_mod}, n_max={n_max}.")
            print(f"    Grid size: n_a={n_a}, n_b={n_b}, n_c={n_c}")
            # print(f"    Grid rescaled: power_a={p_a}, power_b={p_b}, power_c={p_c}")
            print(f"    Energy bins: {energy_bin_num} bins from "
                  f"{energy_threshold:.2e} eV to {energy_max:.2e} eV")

            print("\n    Generate grids and calculating form factor...")
            start_time = time.time()

        # Prepare grid points and basis function values
        lm_list = [(l, m) for l in range(0, l_max + 1, l_mod) for m in range(-l, l + 1)]
        if log_wavelet:
            q_min = energy_threshold / (const.VESC + const.VE)
            eps = q_min / q_max
            self.info["log_wavelet_eps"] = eps
            q_xyz_list, y_lm_vals, jacob_vals = utility.gen_log_mesh_ylm_jacob(
                lm_list=lm_list, u_max=q_max, n_a=n_a, n_b=n_b, n_c=n_c, eps=eps
            )
        else:
            eps = 0.
            q_xyz_list, y_lm_vals, jacob_vals = utility.gen_mesh_ylm_jacob(
                lm_list=lm_list, u_max=q_max, n_a=n_a, n_b=n_b, n_c=n_c  #, p_a
            )

        # Calculate form factor on grid
        form_factor_bin_vals = physics.form_factor(
            q_xyz_list,
            energy_threshold,
            energy_bin_width,
            energy_max,
            n_DW_params,
            phonopy_params,
            c_dict,
            phonon_file,
        ).reshape(energy_bin_num, n_a, n_b, n_c)
        del q_xyz_list

        if verbose:
            end_time = time.time()
            print(f"    Form factor calculation completed in "
                  f"{end_time - start_time:.2f} seconds.")
            print("\n    Projecting form factor onto basis functions...")

        # Project form factor onto basis functions
        for i_bin in range(energy_bin_num):

            self.fnlms[i_bin] = Fnlm(
                l_max=l_max, l_mod=self.l_mod, n_max=n_max, info=self.info
            )

            if verbose:
                if i_bin % (energy_bin_num // 5 + 1) == 0:
                    print(f"      Projecting energy bin {i_bin}/{energy_bin_num-1}...")

            f_nlm = utility.proj_get_f_nlm(
                n_max=n_max, lm_list=lm_list,
                func_vals=form_factor_bin_vals[i_bin, :, :, :], 
                y_lm_vals=y_lm_vals, jacob_vals=jacob_vals,
                n_a=n_a, # p_a,
                log_wavelet=log_wavelet, eps=eps
            )

            self.fnlms[i_bin].f_lm_n = np.zeros(
                (self.fnlms[i_bin].get_lm_index(l_max, l_max)+1, n_max + 1,),
                dtype=float,
            )
            for l in range(0, l_max + 1, l_mod):
                for m in range(-l, l + 1):
                    idx_lm = self.fnlms[i_bin].get_lm_index(l, m)
                    for n in range(n_max + 1):
                        self.fnlms[i_bin].f_lm_n[idx_lm, n] = (
                            f_nlm.get((n, l, m), 0.0)
                        )

        if verbose:
            print("    Projection completed.")
            end_total_time = time.time()
            print(f"    Total projection time: "
                  f"{end_total_time - start_total_time:.2f} seconds.")


class McalI:
    """
    McalI class.

    Parameters
    ----------
    physics_params : dict
        Physics parameters required for McalI projection.
            - fdm: tuple, Dark matter form factor parameters (a,b) with
                   F_DM(q,v) = (q/q0)**a * (v/v0)**b
            - q0_fdm: (optional) reference momentum for FDM grid.
            - energy: energy transfer.
            - mass_dm: dark matter mass.
            - mass_sm: (optional) standard model particle mass.

    numerics_params : dict
        Numerical parameters required for McalI projection.
            - l_max: maximum angular momentum quantum number.
            - l_mod: (optional) denotes parity of l values (1 for all, 2 for even only).
            - nv_list: list of velocity radial quantum numbers.
            - nq_list: list of momentum radial quantum numbers.
            - v_max: reference velocity scale.
            - q_max: reference momentum scale.
            - log_wavelet_q: (optional) whether to use log wavelet mesh for q.
            - eps_q: (optional) epsilon parameter for log wavelet mesh.
    """
    def __init__(self, physics_params, numerics_params):
        self.l_max = numerics_params.get("l_max", -1)
        self.l_mod = numerics_params.get("l_mod", 1)
        self.nv_max = numerics_params.get("nv_max", -1)
        self.nq_max = numerics_params.get("nq_max", -1)
        self.v_max = numerics_params.get("v_max", 1.0)
        self.q_max = numerics_params.get("q_max", 1.0)

        self.fdm = physics_params.get("fdm", (0, 0))
        self.q0_fdm = physics_params.get("q0_fdm", const.Q_BOHR)
        self.energy = physics_params.get("energy", 0.0)
        self.mass_dm = physics_params.get("mass_dm", 1.0)
        self.mass_sm = physics_params.get("mass_sm", const.M_NUCL)

        self.log_wavelet_q = numerics_params.get("log_wavelet_q", False)
        self.eps_q = numerics_params.get("eps_q", 1.0)

        if self.l_mod not in [1, 2]:
            raise ValueError("l_mod must be either 1 (all l) or 2 (even l only).")

        if self.log_wavelet_q and (self.eps_q <= 0.0 or self.eps_q > 1.0):
            raise ValueError("eps_q must be in (0, 1) for log wavelet basis in q.")

        self.mcalI = np.zeros(
            (self.l_max//self.l_mod + 1, self.nv_max + 1, self.nq_max + 1), dtype=float
        )
        self.info = {}

    def export_hdf5(
        self, filename, groupname, dataname="data", 
        write_info=True, verbose=True, dtype=np.float64
    ):
        if not filename.endswith(".hdf5"):
            filename += ".hdf5"
        with h5py.File(filename, "a") as h5f:

            grp = h5f.require_group(groupname)
            if dataname in grp:
                del grp[dataname]
            dset = grp.create_dataset(dataname, data=self.mcalI, dtype=dtype)

            if write_info:
                dset.attrs["l_max"] = self.l_max
                dset.attrs["l_mod"] = self.l_mod
                dset.attrs["nv_max"] = self.nv_max
                dset.attrs["nq_max"] = self.nq_max
                dset.attrs["v_max"] = self.v_max
                dset.attrs["q_max"] = self.q_max

                dset.attrs["fdm"] = self.fdm
                dset.attrs["q0_fdm"] = self.q0_fdm
                dset.attrs["energy"] = self.energy
                dset.attrs["mass_dm"] = self.mass_dm
                dset.attrs["mass_sm"] = self.mass_sm

                dset.attrs["log_wavelet_q"] = self.log_wavelet_q
                dset.attrs["eps_q"] = self.eps_q

                for key, value in self.info.items():
                    dset.attrs[key] = value

        if verbose:
            print(f"    McalI data written to {C_GREEN}{filename}{C_RESET} "
                  f"in group {C_CYAN}{groupname}/{dataname}{C_RESET}.")

    def import_hdf5(self, filename, groupname, dataname="data", verbose=True):
        if not filename.endswith(".hdf5"):
            filename += ".hdf5"
        with h5py.File(filename, "r") as h5f:
            grp = h5f[groupname]
            dset = grp[dataname]
            self.mcalI = dset[()]

            keys_to_load = [
                "l_max",
                "l_mod",
                "nv_max",
                "nq_max",
                "v_max",
                "q_max",
                "fdm",
                "q0_fdm",
                "energy",
                "mass_dm",
                "mass_sm",
                "log_wavelet_q",
                "eps_q",
            ]
            for key in keys_to_load:
                if key in dset.attrs:
                    setattr(self, key, dset.attrs[key])
                else:
                    raise KeyError(f"Key '{key}' not found in HDF5 attributes.")
                
            self.info = {
                key: dset.attrs[key] for key in dset.attrs 
                if key not in keys_to_load
            }

        if verbose:
            print(f"    McalI data read from {C_GREEN}{filename}{C_RESET} "
                  f"in group {C_CYAN}{groupname}/{dataname}{C_RESET}.")

    def project(self, verbose=False):

        if verbose:
            print("\n    Starting projection of McalI onto basis functions...")
            start_time = time.time()

        l_max = self.l_max
        l_mod = self.l_mod
        nv_max = self.nv_max
        nq_max = self.nq_max
        v_max = self.v_max
        q_max = self.q_max
        fdm = self.fdm
        q0_fdm = self.q0_fdm
        energy = self.energy
        mass_dm = self.mass_dm
        mass_sm = self.mass_sm
        log_wavelet_q = self.log_wavelet_q
        eps_q = self.eps_q
        
        if verbose:
            print(f"    Using FDM form factor parameters: fdm={fdm}, q0_fdm={q0_fdm}, v0_fdm={1.0}.")
            print(f"    Using DM and SM parameters: mass_dm={mass_dm:.2e} eV, mass_sm={mass_sm:.2e} eV, energy={energy:.2e} eV.")
            print(f"    Parameters for wavelets:")
            if log_wavelet_q:
                print(f"    Log wavelet basis in q with eps_q={eps_q:.2e}.")
            else:
                print(f"    Standard (power) wavelet basis in q.")
            print(f"    Reference velocity v_max = {v_max:.2e}.")
            print(f"    Reference momentum q_max = {q_max:.2e} eV.")
            print(f"    Projecting onto basis with l_max={l_max}, l_mod={l_mod}, nv_max={nv_max}, nq_max={nq_max}.")

            print("\n    Calculating McalI matrix coefficients...")

        # Projection
        # for l in range(0, l_max + 1, l_mod):
        #     for idx_nv, nv in enumerate(nv_list):
        #         for idx_nq, nq in enumerate(nq_list):
        #             self.mcalI[l, idx_nv, idx_nq] = self.getI_lvq_analytic((l, nv, nq))
        self.mcalI = self.get_mcalI(l_max, l_mod, nv_max, nq_max)

        if verbose:
            print("    Projection completed.")
            end_time = time.time()
            print(f"    Total projection time: {end_time - start_time:.2f} seconds.")

    def getI_lvq_analytic(self, lnvq, verbose=False):
        """Analytic calculation for I(ell) matrix coefficients.

        Only available for 'tophat' and 'wavelet' bases (so far).

        Arguments:
            lnvq = (ell, nv, nq)
            verbose: whether to print output
        """
        v_max = self.v_max
        q_max = self.q_max
        mass_dm = self.mass_dm
        fdm = self.fdm  # DM-SM scattering form factor index
        # (a, b) = fdm
        q0_fdm = self.q0_fdm  # reference momentum for FDM form factor
        v0_fdm = 1.0  # reference velocity for FDM form factor
        mass_sm = self.mass_sm  # SM particle mass (mElec)
        energy = self.energy  # DM -> SM energy transfer
        log_wavelet_q = self.log_wavelet_q
        eps_q = self.eps_q

        Ilvq = analytic.ilvq_analytic(
            lnvq, v_max, q_max, log_wavelet_q, eps_q,
            fdm, q0_fdm, v0_fdm, 
            mass_dm, mass_sm, energy, verbose=verbose
        )

        # Ilvq = analytic.ilvq_vsdm(
        #     lnvq, v_max, q_max, log_wavelet_q, eps_q,
        #     fdm, q0_fdm, v0_fdm, 
        #     mass_dm, mass_sm, energy, verbose=verbose
        # )

        return Ilvq

    def get_mcalI(self, l_max, l_mod, nv_max, nq_max):

        # l_max = self.l_max
        # l_mod = self.l_mod
        # nv_max = self.nv_max
        # nq_max = self.nq_max
        v_max = self.v_max
        q_max = self.q_max
        fdm = self.fdm
        q0_fdm = self.q0_fdm
        v0_fdm = 1.0  # reference velocity for FDM form factor
        energy = self.energy
        mass_dm = self.mass_dm
        mass_sm = self.mass_sm
        log_wavelet_q = self.log_wavelet_q
        eps_q = self.eps_q

        return analytic.ilvq(
            l_max, l_mod, nv_max, nq_max,
            v_max, q_max, log_wavelet_q, eps_q,
            fdm, q0_fdm, v0_fdm, 
            mass_dm, mass_sm, energy
        )


class BinnedMcalI:
    """
    Binned McalI class.

    Parameters
    ----------
    physics_params : dict
        Physics parameters required for Binned McalI projection.
            - fdm: tuple, Dark matter form factor parameters (a,b) with
                   F_DM(q,v) = (q/q0)**a * (v/v0)**b
            - q0_fdm: (optional) reference momentum for FDM grid.
            - energy_threshold: minimum energy transfer.
            - energy_bin_width: width of each energy bin.
            - mass_dm: dark matter mass.
            - mass_sm: (optional) standard model particle mass.

    numerics_params : dict
        Numerical parameters required for Binned McalI projection.
            - l_max: maximum angular momentum quantum number.
            - l_mod: (optional) denotes parity of l values (1 for all, 2 for even only).
            - nv_list: list of velocity radial quantum numbers.
            - nq_list: list of momentum radial quantum numbers.
            - v_max: reference velocity scale.
            - q_max: reference momentum scale.
            - n_bins: (optional) number of energy bins.
            - log_wavelet_q: (optional) whether to use log wavelet mesh for q.
            - eps_q: (optional) epsilon parameter for log wavelet mesh.
    """
    def __init__(self, physics_params, numerics_params):
        self.l_max = numerics_params.get("l_max", -1)
        self.l_mod = numerics_params.get("l_mod", 1)
        self.nv_max = numerics_params.get("nv_max", -1)
        self.nq_max = numerics_params.get("nq_max", -1)
        self.v_max = numerics_params.get("v_max", 1.0)
        self.q_max = numerics_params.get("q_max", 1.0)

        self.fdm = physics_params.get("fdm", (0, 0))
        self.q0_fdm = physics_params.get("q0_fdm", const.Q_BOHR)
        self.n_bins = numerics_params.get("n_bins", 0)
        self.energy_threshold = physics_params.get("energy_threshold", 0.0)
        self.energy_bin_width = physics_params.get("energy_bin_width", 0.0)
        self.mass_dm = physics_params.get("mass_dm", 1.0)
        self.mass_sm = physics_params.get("mass_sm", const.M_NUCL)

        self.log_wavelet_q = numerics_params.get("log_wavelet_q", False)
        self.eps_q = numerics_params.get("eps_q", 1.0)

        if self.l_mod not in [1, 2]:
            raise ValueError("l_mod must be either 1 (all l) or 2 (even l only).")

        if self.log_wavelet_q and (self.eps_q <= 0.0 or self.eps_q > 1.0):
            raise ValueError("eps_q must be in (0, 1) for log wavelet basis in q.")

        self.mcalIs = {}  # to be filled after projection
        self.info = {}
    
    def _check_consistency(self):
        for idx_bin, mcalI in self.mcalIs.items():
            if mcalI.l_max != self.l_max:
                raise ValueError(f"Inconsistent l_max in bin {idx_bin}.")
            if mcalI.l_mod != self.l_mod:
                raise ValueError(f"Inconsistent l_mod in bin {idx_bin}.")
            if mcalI.nv_max != self.nv_max:
                raise ValueError(f"Inconsistent nv_max in bin {idx_bin}.")
            if mcalI.nq_max != self.nq_max:
                raise ValueError(f"Inconsistent nq_max in bin {idx_bin}.")
            if mcalI.v_max != self.v_max:
                raise ValueError(f"Inconsistent v_max in bin {idx_bin}.")
            if mcalI.q_max != self.q_max:
                raise ValueError(f"Inconsistent q_max in bin {idx_bin}.")
            if any(mcalI.fdm) != any(self.fdm):
                raise ValueError(f"Inconsistent fdm in bin {idx_bin}.")
            if mcalI.q0_fdm != self.q0_fdm:
                raise ValueError(f"Inconsistent q0_fdm in bin {idx_bin}.")
            if mcalI.mass_dm != self.mass_dm:
                raise ValueError(f"Inconsistent mass_dm in bin {idx_bin}.")
            if mcalI.mass_sm != self.mass_sm:
                raise ValueError(f"Inconsistent mass_sm in bin {idx_bin}.")
            if mcalI.log_wavelet_q != self.log_wavelet_q:
                raise ValueError(f"Inconsistent log_wavelet_q in bin {idx_bin}.")
            if mcalI.eps_q != self.eps_q:
                raise ValueError(f"Inconsistent eps_q in bin {idx_bin}.")

    def export_hdf5(
        self, filename, groupname, dataname="data", 
        write_sub_info=True, verbose=True, dtype=np.float64
    ):
        if not filename.endswith(".hdf5"):
            filename += ".hdf5"
        with h5py.File(filename, "a") as h5f:
            grp = h5f.require_group(groupname)

            grp.attrs["n_bins"] = self.n_bins
            grp.attrs["l_max"] = self.l_max
            grp.attrs["l_mod"] = self.l_mod
            grp.attrs["nv_max"] = self.nv_max
            grp.attrs["nq_max"] = self.nq_max
            grp.attrs["v_max"] = self.v_max
            grp.attrs["q_max"] = self.q_max

            grp.attrs["fdm"] = self.fdm
            grp.attrs["q0_fdm"] = self.q0_fdm
            grp.attrs["energy_threshold"] = self.energy_threshold
            grp.attrs["energy_bin_width"] = self.energy_bin_width
            grp.attrs["mass_dm"] = self.mass_dm
            grp.attrs["mass_sm"] = self.mass_sm

            grp.attrs["log_wavelet_q"] = self.log_wavelet_q
            grp.attrs["eps_q"] = self.eps_q

            for key, value in self.info.items():
                grp.attrs[key] = value

            for idx_bin, mcalI in self.mcalIs.items():
                grp.require_group(f"bin_{idx_bin}")
                mcalI.export_hdf5(
                    filename,
                    f"{groupname}/bin_{idx_bin}",
                    dataname,
                    write_info=write_sub_info,
                    verbose=False,
                    dtype=dtype,
                )

        if verbose:
            print(
                f"    BinnedMcalI data written to {C_GREEN}{filename}{C_RESET} "
                f"in group {C_CYAN}{groupname}/bin_*{C_RESET}.",
            )

    def import_hdf5(self, filename, groupname, dataname="data", verbose=True):
        if not filename.endswith(".hdf5"):
            filename += ".hdf5"
        with h5py.File(filename, "r") as h5f:
            grp = h5f[groupname]

            keys_to_load = [
                "n_bins",
                "l_max",
                "l_mod",
                "nv_max",
                "nq_max",
                "v_max",
                "q_max",
                "fdm",
                "q0_fdm",
                "energy_threshold",
                "energy_bin_width",
                "mass_dm",
                "mass_sm",
                "log_wavelet_q",
                "eps_q",
            ]
            for key in keys_to_load:
                if key in grp.attrs:
                    setattr(self, key, grp.attrs[key])
                else:
                    raise KeyError(f"Key '{key}' not found in HDF5 attributes.")
                
            self.info = {
                key: grp.attrs[key] for key in grp.attrs if key not in keys_to_load
            }

            self.mcalIs = {}
            for idx_bin in range(self.n_bins):
                grp.require_group(f"bin_{idx_bin}")
                mcalI = McalI(physics_params={}, numerics_params={})
                mcalI.import_hdf5(
                    filename, f"{groupname}/bin_{idx_bin}", dataname, verbose=False
                )
                self.mcalIs[idx_bin] = mcalI

        self._check_consistency()

        if verbose:
            print(
                f"    BinnedMcalI data read from {C_GREEN}{filename}{C_RESET} "
                f"in group {C_CYAN}{groupname}bin_*{C_RESET}.",
            )

    def project(self, verbose=False):

        if verbose:
            print("\n    Starting projection of McalI onto basis functions...")
            start_time = time.time()

        physics_params_keys = [
            "fdm",
            "q0_fdm",
            "mass_dm",
            "mass_sm",
        ]
        numerics_params_keys = [
            "l_max",
            "l_mod",
            "nv_max",
            "nq_max",
            "v_max",
            "q_max",
            "log_wavelet_q",
            "eps_q",
        ]
        physics_params = {
            key: self.__dict__[key] for key in physics_params_keys
        }
        numerics_params = {
            key: self.__dict__[key] for key in numerics_params_keys
        }

        if verbose:
            print(f"    Using FDM form factor parameters: fdm={self.fdm}, q0_fdm={self.q0_fdm:.2e} eV, v0_fdm={1.0}.")
            print(f"    Using DM and SM parameters: mass_dm={self.mass_dm:.2e} eV, mass_sm={self.mass_sm:.2e} eV")
            print(f"    In total {self.n_bins} energy bins starting from")
            print(f"    energy_threshold={self.energy_threshold:.2e} eV, energy_bin_width={self.energy_bin_width:.2e} eV.\n")
            print(f"    Parameters for wavelets:")
            if self.log_wavelet_q:
                print(f"    Log wavelet basis in q with eps_q={self.eps_q:.2e}.")
            else:
                print(f"    Standard (linear) wavelet basis in q.")
            print(f"    Reference velocity v_max = {self.v_max:.2e}.")
            print(f"    Reference momentum q_max = {self.q_max:.2e} eV.")
            print(f"    Projecting onto basis with l_max={self.l_max}, nv_max={self.nv_max}, nq_max={self.nq_max}.")

            print("\n    Calculating McalI matrix coefficients...")

        n_bins = self.n_bins

        for idx_bin in range(n_bins):
            if verbose:
                if idx_bin % (n_bins // 5 + 1) == 0:
                    print(f"        Projecting energy bin {idx_bin}/{n_bins-1}...")

            energy = (
                self.energy_threshold
                + (idx_bin + 0.5) * self.energy_bin_width
            )
            physics_params["energy"] = energy

            mcalI = McalI(physics_params, numerics_params)
            mcalI.project(verbose=False)
            self.mcalIs[idx_bin] = mcalI

        if verbose:
            end_time = time.time()
            print(f"    Total projection time: {end_time - start_time:.2f} seconds.")
