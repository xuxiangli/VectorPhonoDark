import numpy as np
import numba
import time
import os
import csv
import h5py
from functools import reduce

import phonopy
import vsdm

from . import constants as const
from . import basis_funcs
from . import projection as proj
from .utility import Color
from . import phonopy_funcs
from . import physics


def get_intersection_index(**lists):
        """
        Get the indices of common elements in multiple lists.

        Args:
            base_list (list): The base list to compare against.
            **lists: Arbitrary number of lists to find common elements with.

        Returns:
            np.ndarray: Array of indices in each list corresponding to the common elements.
                shape: (number of lists, number of common elements)
        """
        target_arrays = [np.asarray(l) for l in lists.values()]
        if not target_arrays:
            return np.array([])

        common_elements = reduce(np.intersect1d, target_arrays)
        if common_elements.size == 0:
            return np.empty((len(lists), 0), dtype=int)
        
        indices = np.vstack([np.searchsorted(arr, common_elements) 
                                   for arr in target_arrays])
        return indices


class Fnlm:
    def __init__(self):
        self.l_max = -1
        self.l_mod = 1
        self.n_list = np.array([])
        self.f_lm_n = np.array([])
        self.info = {}
    
    def get_lm_index(self, l, m):
        if abs(m) > l or l > self.l_max or l % self.l_mod != 0:
            raise ValueError("Invalid (l, m) values.")
        return l**2 + l + m
    
    def get_n_index(self, n):
        if n not in self.n_list:
            raise ValueError("n value not in n_list.")
        if not hasattr(self, 'n_index_map') or n not in self.n_index_map:
            self.n_index_map = {val: idx for idx, val in enumerate(self.n_list)}
        return self.n_index_map[n]

    def export_csv(self, filename, write_info=True, verbose=True):
        if not filename.endswith('.csv'):
            filename += '.csv'
        makeHeader = not os.path.exists(filename)
        with open(filename, mode='a') as file:
            writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            if makeHeader:
                if write_info:
                    bparams = [r'#'] + [str(key)+':'+str(value) 
                                        for key, value in self.info.items()]
                    writer.writerow(bparams)
                header = [r'#n', 'l', 'm', 'f_lm_n']
                writer.writerow(header)
            for l in range(self.l_max + 1):
                for m in range(-l, l + 1):
                    idx_lm = self.get_lm_index(l, m)
                    for idx_n, n in enumerate(self.n_list):
                        row = [n, l, m, self.f_lm_n[idx_lm, idx_n]]
                        writer.writerow(row)
        
        if verbose:
            print(f"    Fnlm data written to {Color.GREEN}{filename}{Color.RESET}.")

    def import_csv(self, filename, verbose=True):
        if not filename.endswith('.csv'):
            filename += '.csv'
        # read header and get l_max, n_list, l_mod first
        with open(filename, mode='r') as file:
            reader = csv.reader(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            data = {}
            n_list = set()
            l_list = set()
            for row in reader:
                if row[0].startswith('#'):
                    if len(row) > 1 and ':' in row[1]:
                        for item in row[1:]:
                            key, value = item.split(':', 1)
                            if key == 'l_max':
                                self.l_max = int(value)
                            elif key == 'l_mod':
                                self.l_mod = int(value)
                            elif key == 'n_list':
                                n_items = value.strip('[]').split(',')
                                self.n_list = np.array([int(n) for n in n_items])
                            else:
                                self.info[key] = value
                    continue
                else:
                    n, l, m = int(row[0]), int(row[1]), int(row[2])
                    value = float(row[3])
                    n_list.add(n)
                    l_list.add(l)
                    data[(n, l, m)] = value
            
            if self.l_max == -1:
                self.l_max = max(l_list)
            if self.l_mod == 1:
                self.l_mod = 2 if all(l % 2 == 0 for l in l_list) else 1
            if len(self.n_list) == 0:
                self.n_list = np.array(sorted(n_list))
            
            self.f_lm_n = np.zeros((
                self.get_lm_index(self.l_max, self.l_max)+1, len(self.n_list)
            ), dtype=float)
            for (n, l, m), value in data.items():
                idx_lm = self.get_lm_index(l, m)
                idx_n = self.get_n_index(n)
                self.f_lm_n[idx_lm, idx_n] = value

        if verbose:
            print(f"    Fnlm data read from {Color.GREEN}{filename}{Color.RESET}.")


    def export_hdf5(self, filename, groupname, dataname='data', 
                   write_info=True, verbose=True):
        if not filename.endswith('.hdf5'):
            filename += '.hdf5'
        with h5py.File(filename, 'a') as h5f:
            grp = h5f.require_group(groupname)
            if dataname in grp:
                del grp[dataname]
            dset = grp.create_dataset(dataname, data=self.f_lm_n)
            dset.attrs['l_max'] = self.l_max
            dset.attrs['l_mod'] = self.l_mod
            dset.attrs['n_list'] = self.n_list
            if write_info:
                for key, value in self.info.items():
                    dset.attrs[key] = value
        if verbose:
            print(
                f"    Fnlm data written to {Color.GREEN}{filename}{Color.RESET}",
                f"in group {Color.CYAN}{groupname}/{dataname}{Color.RESET}."
            )

    def import_hdf5(self, filename, groupname, dataname='data', verbose=True):
        if not filename.endswith('.hdf5'):
            filename += '.hdf5'
        with h5py.File(filename, 'r') as h5f:
            grp = h5f[groupname]
            dset = grp[dataname]
            self.f_lm_n = dset[()]
            self.l_max = dset.attrs['l_max']
            self.l_mod = dset.attrs['l_mod']
            self.n_list = dset.attrs['n_list']
            self.info = {key: dset.attrs[key] for key in dset.attrs 
                         if key not in ['l_max', 'l_mod', 'n_list']}
            
        if verbose:
            print(
                f"    Fnlm data read from {Color.GREEN}{filename}{Color.RESET}",
                f"    in group {Color.CYAN}{groupname}/{dataname}{Color.RESET}."
            )


class BinnedFnlm:
    def __init__(self):
        self.n_bins = 0
        self.l_max = -1
        self.l_mod = 1
        self.n_list = np.array([])
        self.fnlms = {}
        self.info = {}
    
    def export_csv(self, filename, write_info=True, verbose=True):
        if filename.endswith('.csv'):
            filename = filename[:-4]
        for idx_bin in range(self.n_bins):
            bin_filename = filename + '_bin_' + str(idx_bin) + '.csv'
            self.fnlms[idx_bin].export_csv(bin_filename, write_info=write_info, 
                                           verbose=False)
        if verbose:
            print(f"    BinnedFnlm data written to {Color.GREEN}{filename}_bin_*.csv{Color.RESET} files.")
    
    def import_csv(self, filename, verbose=True):
        if filename.endswith('.csv'):
            filename = filename[:-4]
        
        self.fnlms = {}
        idx_bin = 0
        while True:
            try:
                bin_filename = filename + '_bin_' + str(idx_bin) + '.csv'
                fnlm = Fnlm()
                fnlm.import_csv(bin_filename, verbose=False)
                self.fnlms[idx_bin] = fnlm
                self.info = fnlm.info
                idx_bin += 1
            except FileNotFoundError:
                break
        self.n_bins = idx_bin

        if self.n_bins == 0:
            raise ValueError("No bin files found for BinnedFnlm import.")

        # check consistency of l_max, n_list, l_mod
        l_max_list = [fnlm.l_max for fnlm in self.fnlms.values()]
        n_list_list = [fnlm.n_list for fnlm in self.fnlms.values()]
        l_mod_list = [fnlm.l_mod for fnlm in self.fnlms.values()]
        if not all(l_max == l_max_list[0] for l_max in l_max_list):
            raise ValueError("Inconsistent l_max among bins.")
        else:
            self.l_max = l_max_list[0]
        if not all(np.array_equal(n_list, n_list_list[0]) for n_list in n_list_list):
            raise ValueError("Inconsistent n_list among bins.")
        else:
            self.n_list = n_list_list[0]
        if not all(l_mod == l_mod_list[0] for l_mod in l_mod_list):
            raise ValueError("Inconsistent l_mod among bins.")
        else:
            self.l_mod = l_mod_list[0]

        if verbose:
            print(f"    BinnedFnlm data read from {Color.GREEN}{filename}_bin_*.csv{Color.RESET} files.") 

    def export_hdf5(self, filename, groupname, dataname='data', 
                   write_info=True, verbose=True):
        if not filename.endswith('.hdf5'):
            filename += '.hdf5'
        with h5py.File(filename, 'a') as h5f:
            grp = h5f.require_group(groupname)
            grp.attrs['n_bins'] = self.n_bins
            grp.attrs['l_max'] = self.l_max
            grp.attrs['n_list'] = self.n_list
            for key, value in self.info.items():
                grp.attrs[key] = value
            for idx_bin, fnlm in self.fnlms.items():
                bin_grp = grp.require_group(f'bin_{idx_bin}')
                fnlm.export_hdf5(filename, f'{groupname}/bin_{idx_bin}', dataname, 
                                write_info=write_info, verbose=False)
        
        if verbose:
            print(
                f"    BinnedFnlm data written to {Color.GREEN}{filename}{Color.RESET}",
                f"    in group {Color.CYAN}{groupname}/bin_*{Color.RESET}."
            )

    def import_hdf5(self, filename, groupname, dataname='data', verbose=True):
        if not filename.endswith('.hdf5'):
            filename += '.hdf5'
        with h5py.File(filename, 'r') as h5f:
            grp = h5f[groupname]
            self.n_bins = grp.attrs['n_bins']
            self.l_max = grp.attrs['l_max']
            self.n_list = grp.attrs['n_list']
            self.info = {key: grp.attrs[key] for key in grp.attrs 
                         if key not in ['n_bins', 'l_max', 'n_list']}
            self.fnlms = {}
            for idx_bin in range(self.n_bins):
                bin_grp = grp[f'bin_{idx_bin}']
                fnlm = Fnlm()
                fnlm.import_hdf5(filename, f'{groupname}/bin_{idx_bin}', dataname, verbose=False)
                self.fnlms[idx_bin] = fnlm

        if verbose:
            print(
                f"    BinnedFnlm data read from {Color.GREEN}{filename}{Color.RESET}",
                 f"    in group {Color.CYAN}{groupname}/bin_*{Color.RESET}."
            )


class VDF(Fnlm):
    """
    Velocity Distribution Function class.

    Args:
        physics_params: 
            dict: Physics parameters required for VDF projection.
                - vdf: the velocity distribution function.
                - vdf_params: parameters for the velocity distribution function.
        numerics_params: 
            dict: Numerical parameters required for VDF projection.
                - v_max: reference velocity scale.
                - l_max: maximum angular momentum quantum number.
                - n_list: list of radial quantum numbers.
                - n_grid: number of grid points for velocity discretization.
    """
    def __init__(self, physics_params, numerics_params):
        super().__init__()
        self.l_max = numerics_params.get('l_max', -1)
        self.l_mod = numerics_params.get('l_mod', 1)
        self.n_list = numerics_params.get('n_list', [])

        self.vdf = physics_params.get('vdf', None)
        self.vdf_params = physics_params.get('vdf_params', {})

        self.v_max = numerics_params.get('v_max', 1.)
        self.n_grid = numerics_params.get('n_grid', (32, 25, 25))
        
        # Store additional info
        for key, value in self.vdf_params.items():
            self.info['vdf_param_'+key] = value
        self.info['v_max'] = self.v_max
        self.info['n_grid'] = self.n_grid

        if 'model' in physics_params:
            self.info['model'] = physics_params['model']
    
    def import_csv(self, filename, verbose=True):
        super().import_csv(filename, verbose=verbose)
        self.v_max = float(self.info.get('v_max', 1.))
        self.info['v_max'] = self.v_max

    def import_hdf5(self, filename, groupname, dataname='data', verbose=True):
        super().import_hdf5(filename, groupname, dataname, verbose)
        self.v_max = float(self.info.get('v_max', 1.))
        self.info['v_max'] = self.v_max
    
    def project(self, params, verbose=False):
        
        if verbose:
            print("\n    Starting projection of velocity distribution function onto basis functions...")
            start_total_time = time.time()

        v_max = params.get('v_max', self.v_max)
        vdf = params.get('vdf', self.vdf)
        vdf_params = params.get('vdf_params', self.vdf_params)
        
        l_max = params.get('l_max', self.l_max)
        n_list = params.get('n_list', self.n_list)
        n_list = np.array(n_list)

        n_a, n_b, n_c = params.get('n_grid', self.n_grid)
        p_a, p_b, p_c = params.get('power_grid', (1, 1, 1))

        assert p_b == 1 and p_c == 1, "Currently only power 1 is supported for angular grids."

        if verbose:
            if 'model' in self.info:
                print(f"    Using VDF model: {self.info['model']}")
            print(f"    Parameters for VDF: {vdf_params}")
            print(f"    Reference velocity v_max = {v_max:.2e}.")
            print(f"    Projecting onto basis with l_max={l_max}, n_max={max(n_list)}.")
            print(f"    Grid size: n_a={n_a}, n_b={n_b}, n_c={n_c}")
            print(f"    Grid rescaled: power_a={p_a}, power_b={p_b}, power_c={p_c}")

            print("\n    Generate grids and calculating form factor...")
            start_time = time.time()

        # Prepare grid points and basis function values
        lm_list = [(l, m) for l in range(l_max + 1) for m in range(-l, l+1)]
        v_xyz_list, y_lm_vals, jacob_vals = proj.generate_mesh_ylm_jacob(
            lm_list, v_max, n_a, n_b, n_c, p_a, p_b, p_c
        )

        # Calculate vdf on grid
        vdf_vals = np.array(
            [vdf(v_vec, **vdf_params) for v_vec in v_xyz_list]
        ).reshape(n_a, n_b, n_c)
        del v_xyz_list

        if verbose:
            end_time = time.time()
            print(
                f"    VDF calculation completed in {end_time - start_time:.2f} seconds.")
            print("\n    Projecting VDF onto basis functions and saving results...")

        # Project vdf onto basis functions
        f_nlm = proj.proj_get_f_nlm(
            n_list, lm_list,
            vdf_vals, y_lm_vals, jacob_vals,
            'haar', n_a, p_a, verbose=verbose
        )
        del vdf_vals, y_lm_vals, jacob_vals

        # Store results
        self.v_max = v_max
        self.info['v_max'] = self.v_max
        self.l_max = l_max
        self.n_list = n_list
        self.f_lm_n = np.zeros(
            (self.get_lm_index(l_max, l_max)+1, len(n_list)), dtype=float
        )
        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                idx_lm = self.get_lm_index(l, m)
                for idx_n, n in enumerate(n_list):
                    self.f_lm_n[idx_lm, idx_n] = f_nlm.get((n, l, m), 0.0)

        if verbose:
            print("    Projection completed.")
            end_total_time = time.time()
            print(f"    Total projection time: {end_total_time - start_total_time:.2f} seconds.")


class FormFactor(BinnedFnlm):
    """
    Form Factor class.

    Args:
        physics_params: 
            dict: Physics parameters required for Form Factor projection.
                - energy_threshold: minimum energy transfer.
                - energy_bin_width: width of each energy bin.
                - energy_max_factor: factor to determine maximum energy.
        numerics_params: 
            dict: Numerical parameters required for Form Factor projection.
                - q_max: reference momentum scale.
                - l_max: maximum angular momentum quantum number.
                - n_list: list of radial quantum numbers.
                - n_grid: number of grid points for momentum discretization.
                - special_mesh: whether to use special mesh and wavelets.
                - q_cut: whether to compute q_cut from Debye-Waller factor.
    """
    def __init__(self, physics_params, numerics_params):
        super().__init__()
        self.l_max = numerics_params.get('l_max', -1)
        self.l_mod = numerics_params.get('l_mod', 1)
        self.n_list = numerics_params.get('n_list', [])

        self.energy_threshold = physics_params.get('energy_threshold', 1e-3)
        self.energy_bin_width = physics_params.get('energy_bin_width', 1e-3)
        self.energy_max_factor = physics_params.get('energy_max_factor', 4.0)

        self.q_max = numerics_params.get('q_max', 1.)
        self.n_grid = numerics_params.get('n_grid', (32, 25, 25))
        self.special_mesh = numerics_params.get('special_mesh', False)
        self.q_cut = numerics_params.get('q_cut', False)

        # Store additional info
        self.info['energy_threshold'] = self.energy_threshold
        self.info['energy_bin_width'] = self.energy_bin_width
        self.info['q_max'] = self.q_max
        self.info['n_grid'] = self.n_grid
        self.info['special_mesh'] = self.special_mesh
        self.info['q_cut'] = self.q_cut

        if 'model' in physics_params:
            self.info['model'] = physics_params['model']
    
    def import_csv(self, filename, verbose=True):
        super().import_csv(filename, verbose=verbose)
        self.q_max = float(self.info.get('q_max', 1.))
        self.info['q_max'] = self.q_max
        for fnlm in self.fnlms.values():
            fnlm.q_max = self.q_max
            fnlm.info['q_max'] = self.q_max

    def import_hdf5(self, filename, groupname, dataname='data', verbose=True):
        super().import_hdf5(filename, groupname, dataname, verbose=verbose)
        self.q_max = float(self.info.get('q_max', 1.))
        self.info['q_max'] = self.q_max
        for fnlm in self.fnlms.values():
            fnlm.q_max = self.q_max
            fnlm.info['q_max'] = self.q_max

    def project(self, params, verbose=False):
        
        if verbose:
            print("\n    Starting projection of form factor onto basis functions...")
            start_total_time = time.time()

        # Phonon data
        phonon_file, phonopy_params, c_dict, n_DW_params = self._get_phonon_data(params)

        l_max = params.get('l_max', self.l_max)
        n_list = params.get('n_list', self.n_list)
        n_list = np.array(n_list)

        n_a, n_b, n_c = params.get('n_grid', self.n_grid)
        p_a, p_b, p_c = params.get('power_grid', (1, 1, 1))
        special_mesh = params.get('special_mesh', self.special_mesh)

        assert p_b == 1 and p_c == 1, "Currently only power 1 is supported for angular grids."

        energy_threshold = params.get('energy_threshold', self.energy_threshold)
        energy_bin_width = params.get('energy_bin_width', self.energy_bin_width)
        energy_max_factor = params.get('energy_max_factor', self.energy_max_factor)
        energy_max = physics.get_energy_max(phonon_file, factor=energy_max_factor)
        energy_bin_num = int((energy_max - energy_threshold)/energy_bin_width) + 1
        self.n_bins = energy_bin_num
        self.info['n_bins'] = self.n_bins

        q_max = params.get('q_max', self.q_max)
        q_cut_option = params.get('q_cut', self.q_cut)
        q_max = physics.get_q_max(q_max=q_max, q_cut_option=q_cut_option,
                                  phonon_file=phonon_file,
                                  atom_masses=phonopy_params['atom_masses'],
                                  verbose=verbose)

        if verbose:
            if 'material' in self.info:
                print(f"    Material: {self.info['material']}")
            print(f"    Reference momentum q_max = {q_max:.2e} eV.")
            print(f"    Projecting onto basis with l_max={l_max}, n_max={max(n_list)}.")
            print(f"    Grid size: n_a={n_a}, n_b={n_b}, n_c={n_c}")
            print(f"    Grid rescaled: power_a={p_a}, power_b={p_b}, power_c={p_c}")
            print(f"    Energy bins: {energy_bin_num} bins from {energy_threshold:.2e} eV to {energy_max:.2e} eV")

            print("\n    Generate grids and calculating form factor...")
            start_time = time.time()

        # Prepare grid points and basis function values
        lm_list = [(l, m) for l in range(l_max + 1) for m in range(-l, l+1)]
        q_xyz_list, y_lm_vals, jacob_vals = proj.generate_mesh_ylm_jacob(
            lm_list, q_max, n_a, n_b, n_c, p_a, p_b, p_c
        )
        if special_mesh:
            q_min = energy_threshold / (const.VESC + const.VE)
            lam_start = basis_funcs.haar_n_to_lam_mu(max(n_list))[0] + 1
            q_xyz_list_exp, jacob_vals_exp, n_list_exp = proj.generate_special_mesh_jacob_nlist(
                q_max, q_min, n_b, n_c, lam_start, verbose=verbose
            )

        # Calculate form factor on grid
        form_factor_bin_vals = physics.form_factor(
            q_xyz_list,
            energy_threshold, energy_bin_width, energy_max,
            n_DW_params, phonopy_params, c_dict, phonon_file
        ).reshape(energy_bin_num, n_a, n_b, n_c)
        del q_xyz_list
        if special_mesh:
            form_factor_bin_vals_exp = physics.form_factor(
                q_xyz_list_exp,
                energy_threshold, energy_bin_width, energy_max,
                n_DW_params, phonopy_params, c_dict, phonon_file
            ).reshape(energy_bin_num, -1, n_b, n_c)
            del q_xyz_list_exp

        if verbose:
            end_time = time.time()
            print(f"    Form factor calculation completed in {end_time - start_time:.2f} seconds.")
            print("\n    Projecting form factor onto basis functions and saving results...")

        # Project form factor onto basis functions
        for i_bin in range(energy_bin_num):

            self.fnlms[i_bin] = Fnlm(info=self.info)

            if verbose:
                if i_bin % (energy_bin_num // 5 + 1) == 0:
                    print(f"      Projecting energy bin {i_bin}/{energy_bin_num-1}...")

            f_nlm = proj.proj_get_f_nlm(
                n_list, lm_list,
                form_factor_bin_vals[i_bin, :, :, :], 
                y_lm_vals, jacob_vals,
                'haar', n_a, p_a, verbose=verbose
            )
            self.fnlms[i_bin].l_max = l_max
            self.fnlms[i_bin].n_list = n_list

            if special_mesh:
                f_nlm_exp = proj.proj_get_f_nlm(
                    n_list_exp, lm_list,
                    form_factor_bin_vals_exp[i_bin, :,:, :], 
                    y_lm_vals, jacob_vals_exp,
                    'haar', len(n_list_exp), p_a, special_mesh=True, verbose=verbose
                )
                self.fnlms[i_bin].n_list = np.concatenate((n_list, n_list_exp))

            self.fnlms[i_bin].f_lm_n = np.zeros((
                self.fnlms[i_bin].get_lm_index(l_max, l_max)+1, 
                len(self.fnlms[i_bin].n_list)
            ), dtype=float)
            for l in range(l_max + 1):
                for m in range(-l, l + 1):
                    idx_lm = self.fnlms[i_bin].get_lm_index(l, m)
                    for idx_n, n in enumerate(n_list):
                        self.fnlms[i_bin].f_lm_n[idx_lm, idx_n] = f_nlm.get((n, l, m), 0.0)
                    if special_mesh:
                        for idx_n_exp, n_exp in enumerate(n_list_exp):
                            idx_n_total = len(n_list) + idx_n_exp
                            self.fnlms[i_bin].f_lm_n[idx_lm, idx_n_total] = f_nlm_exp.get((n_exp, l, m), 0.0)
        
        # Store results
        self.l_max = l_max
        self.n_list = n_list
        self.q_max = q_max
        self.special_mesh = special_mesh

        if verbose:
            print("    Projection completed.")
            end_total_time = time.time()
            print(f"    Total projection time: {end_total_time - start_total_time:.2f} seconds.")


    @staticmethod
    def _get_phonon_data(params):

        material_input = params['material_input']
        physics_model_input = params['physics_model_input']
        numerics_input = params['numerics_input']

        mat_input_mod_name = os.path.splitext(os.path.basename(material_input))[0]
        phys_input_mod_name = os.path.splitext(os.path.basename(physics_model_input))[0]
        num_input_mod_name = os.path.splitext(os.path.basename(numerics_input))[0]

        mat_mod = proj.utility.import_file(mat_input_mod_name,
                                          os.path.join(material_input))
        phys_mod = proj.utility.import_file(phys_input_mod_name,
                                           os.path.join(physics_model_input))
        num_mod = proj.utility.import_file(num_input_mod_name,
                                          os.path.join(numerics_input))
        
        material = mat_mod.material
        c_dict = phys_mod.c_dict

        poscar_path = os.path.join(
            os.path.split(material_input)[0], 'POSCAR'
        )
        force_sets_path = os.path.join(
            os.path.split(material_input)[0], 'FORCE_SETS'
        )
        born_path = os.path.join(
            os.path.split(material_input)[0], 'BORN'
        )

        if os.path.exists(born_path):
            born_exists = True
        else:
            print('  There is no BORN file for '+ material +
                '. PHONOPY calculations will process with .NAC. = FALSE\n')
            born_exists = False

        if born_exists:
            phonon_file = phonopy.load(
                supercell_matrix=mat_mod.mat_properties_dict['supercell_dim'],
                primitive_matrix='auto',
                unitcell_filename=poscar_path,
                force_sets_filename=force_sets_path,
                is_nac=True,
                born_filename=born_path
            )
        else:
            phonon_file = phonopy.load(
                supercell_matrix=mat_mod.mat_properties_dict['supercell_dim'],
                primitive_matrix='auto',
                unitcell_filename=poscar_path,
                force_sets_filename=force_sets_path
            )

        phonopy_params = phonopy_funcs.get_phonon_file_data(phonon_file, born_exists)

        n_DW_params = {
            'n_DW_x': num_mod.numerics_parameters['n_DW_x'],
            'n_DW_y': num_mod.numerics_parameters['n_DW_y'],
            'n_DW_z': num_mod.numerics_parameters['n_DW_z']
        }

        return phonon_file, phonopy_params, c_dict, n_DW_params


class McalI:
    def __init__(self, physics_params, numerics_params):
        self.l_max = numerics_params.get('l_max', -1)
        self.l_mod = numerics_params.get('l_mod', 1)
        self.nv_list = numerics_params.get('nv_list', [])
        self.nq_list = numerics_params.get('nq_list', [])
        self.v_max = numerics_params.get('v_max', 1.)
        self.q_max = numerics_params.get('q_max', 1.)

        self.fdm = physics_params.get('fdm', (0,0))
        self.energy = physics_params.get('energy', 0.0)
        self.mass_dm = physics_params.get('mass_dm', 10**6)
        self.mass_sm = physics_params.get('mass_sm', const.M_NUCL)

        self.mcalI = np.array(
            (self.l_max+1, len(self.nv_list), len(self.nq_list)), dtype=float
        )
        self.info = {}
    
    def export_hdf5(self, filename, groupname, dataname='data',
                   write_info=True, verbose=True):
        if not filename.endswith('.hdf5'):
            filename += '.hdf5'
        with h5py.File(filename, 'a') as h5f:
            grp = h5f.require_group(groupname)
            if dataname in grp:
                del grp[dataname]
            dset = grp.create_dataset(dataname, data=self.mcalI)
            if write_info:
                dset.attrs['l_max'] = self.l_max
                dset.attrs['nv_list'] = self.nv_list
                dset.attrs['nq_list'] = self.nq_list
                dset.attrs['v_max'] = self.v_max
                dset.attrs['q_max'] = self.q_max
                dset.attrs['fdm'] = self.fdm
                dset.attrs['energy'] = self.energy
                dset.attrs['mass_dm'] = self.mass_dm
                dset.attrs['mass_sm'] = self.mass_sm
                for key, value in self.info.items():
                    dset.attrs[key] = value
        
        if verbose:
            print(
                f"    McalI data written to {Color.GREEN}{filename}{Color.RESET}",
                f"    in group {Color.CYAN}{groupname}/{dataname}{Color.RESET}."
            )

    def import_hdf5(self, filename, groupname, dataname='data', verbose=True):
        if not filename.endswith('.hdf5'):
            filename += '.hdf5'
        with h5py.File(filename, 'r') as h5f:
            grp = h5f[groupname]
            dset = grp[dataname]
            self.mcalI = dset[()]

            keys_to_load = ['l_max', 'nv_list', 'nq_list', 'v_max', 'q_max', 
                            'fdm', 'energy', 'mass_dm', 'mass_sm']
            for key in keys_to_load:
                if key in dset.attrs:
                    setattr(self, key, dset.attrs[key])
                else:
                    raise KeyError(f"Key '{key}' not found in HDF5 attributes.")
            self.info = {key: dset.attrs[key] for key in dset.attrs 
                         if key not in keys_to_load}
            
        if verbose:
            print(
                f"    McalI data read from {Color.GREEN}{filename}{Color.RESET}",
                f"    in group {Color.CYAN}{groupname}/{dataname}{Color.RESET}."
            )

    def project(self, params, verbose=False):
        
        if verbose:
            print("\n    Starting projection of McalI onto basis functions...")
            start_time = time.time()

        l_max = params.get('l_max', self.l_max)
        nv_list = params.get('nv_list', self.nv_list)
        nq_list = params.get('nq_list', self.nq_list)
        v_max = params.get('v_max', self.v_max)
        q_max = params.get('q_max', self.q_max)
        fdm = params.get('fdm', self.fdm)
        energy = params.get('energy', self.energy)
        mass_dm = params.get('mass_dm', self.mass_dm)
        mass_sm = params.get('mass_sm', self.mass_sm)

        # Prepare for projection using vsdm
        basis_v = dict(u0=v_max, type='wavelet', uMax=v_max)
        basis_q = dict(u0=q_max, type='wavelet', uMax=q_max)
        dm_model = dict(mX=mass_dm, fdm=fdm, mSM=mass_sm, DeltaE=energy)
        mI = vsdm.McalI(basis_v, basis_q, dm_model,
                        use_gvar=False, do_mcalI=False)

        # Projection
        self.mcalI = np.zeros(
            (l_max + 1, len(nv_list), len(nq_list)), dtype=float
        )
        for l in range(l_max + 1):
            for idx_nv, nv in enumerate(nv_list):
                for idx_nq, nq in enumerate(nq_list):
                    self.mcalI[l, idx_nv, idx_nq] = mI.getI_lvq_analytic((l, nv, nq))

        # Store results
        self.l_max = l_max
        self.nv_list = nv_list
        self.nq_list = nq_list
        self.v_max = v_max
        self.q_max = q_max
        self.fdm = fdm
        self.energy = energy
        self.mass_dm = mass_dm
        self.mass_sm = mass_sm

        if verbose:
            print("    Projection completed.")
            end_time = time.time()
            print(f"    Total projection time: {end_time - start_time:.2f} seconds.")


class BinnedMcalI:
    def __init__(self, physics_params, numerics_params):
        self.n_bins = numerics_params.get('n_bins', 0)
        self.l_max = numerics_params.get('l_max', -1)
        self.l_mod = numerics_params.get('l_mod', 1)
        self.nv_list = numerics_params.get('nv_list', [])
        self.nq_list = numerics_params.get('nq_list', [])
        self.v_max = numerics_params.get('v_max', 1.)
        self.q_max = numerics_params.get('q_max', 1.)

        self.fdm = physics_params.get('fdm', (0,0))
        self.energy_threshold = physics_params.get('energy_threshold', 1e-3)
        self.energy_bin_width = physics_params.get('energy_bin_width', 1e-3)
        self.mass_dm = physics_params.get('mass_dm', 10**6)
        self.mass_sm = physics_params.get('mass_sm', const.M_NUCL)
        
        self.mcalIs = {}  # to be filled after projection
        self.info = {}  # to be filled with relevant info
    
    def export_hdf5(self, filename, groupname, dataname='data',
                   write_sub_info=True, verbose=True):
        if not filename.endswith('.hdf5'):
            filename += '.hdf5'
        with h5py.File(filename, 'a') as h5f:
            grp = h5f.require_group(groupname)
            
            grp.attrs['n_bins'] = self.n_bins
            grp.attrs['l_max'] = self.l_max
            grp.attrs['nv_list'] = self.nv_list
            grp.attrs['nq_list'] = self.nq_list
            grp.attrs['v_max'] = self.v_max
            grp.attrs['q_max'] = self.q_max

            grp.attrs['fdm'] = self.fdm
            grp.attrs['energy_threshold'] = self.energy_threshold
            grp.attrs['energy_bin_width'] = self.energy_bin_width
            grp.attrs['mass_dm'] = self.mass_dm
            grp.attrs['mass_sm'] = self.mass_sm

            for key, value in self.info.items():
                grp.attrs[key] = value
            for idx_bin, mcalI in self.mcalIs.items():
                bin_grp = grp.require_group(f'bin_{idx_bin}')
                mcalI.export_hdf5(filename, f'{groupname}/bin_{idx_bin}', dataname, 
                                 write_info=write_sub_info, verbose=False)
        
        if verbose:
            print(
                f"    BinnedMcalI data written to {Color.GREEN}{filename}{Color.RESET}",
                f"    in group {Color.CYAN}{groupname}/bin_*{Color.RESET}."
            )

    def import_hdf5(self, filename, groupname, dataname='data', verbose=True):
        if not filename.endswith('.hdf5'):
            filename += '.hdf5'
        with h5py.File(filename, 'r') as h5f:
            grp = h5f[groupname]

            keys_to_load = ['n_bins', 'l_max', 'nv_list', 'nq_list', 'v_max', 
                            'q_max', 'fdm', 'energy_threshold', 
                            'energy_bin_width', 'mass_dm', 'mass_sm']
            for key in keys_to_load:
                if key in grp.attrs:
                    setattr(self, key, grp.attrs[key])
                else:
                    raise KeyError(f"Key '{key}' not found in HDF5 attributes.")
            self.info = {key: grp.attrs[key] for key in grp.attrs 
                         if key not in keys_to_load}
            
            self.mcalIs = {}
            for idx_bin in range(self.n_bins):
                bin_grp = grp[f'bin_{idx_bin}']
                mcalI = McalI(physics_params={}, numerics_params={})
                mcalI.import_hdf5(filename, f'{groupname}/bin_{idx_bin}', dataname, verbose=False)
                self.mcalIs[idx_bin] = mcalI

            # check consistency
            for idx_bin, mcalI in self.mcalIs.items():
                if mcalI.l_max != self.l_max:
                    raise ValueError(f"Inconsistent l_max in bin {idx_bin}.")
                if not np.array_equal(mcalI.nv_list, self.nv_list):
                    raise ValueError(f"Inconsistent nv_list in bin {idx_bin}.")
                if not np.array_equal(mcalI.nq_list, self.nq_list):
                    raise ValueError(f"Inconsistent nq_list in bin {idx_bin}.")
                if mcalI.v_max != self.v_max:
                    raise ValueError(f"Inconsistent v_max in bin {idx_bin}.")
                if mcalI.q_max != self.q_max:
                    raise ValueError(f"Inconsistent q_max in bin {idx_bin}.")
        
        if verbose:
            print(
                f"    BinnedMcalI data read from {Color.GREEN}{filename}{Color.RESET}",
                f"    in group {Color.CYAN}{groupname}/bin_*{Color.RESET}."
            )

    def project(self, params, verbose=False):
        
        if verbose:
            print("\n    Starting projection of McalI onto basis functions...")
            start_time = time.time()

        params_copy = params.copy()

        n_bins = params_copy.get('n_bins', self.n_bins)

        if 'l_max' not in params_copy:
            params_copy['l_max'] = self.l_max
        if 'nv_list' not in params_copy:
            params_copy['nv_list'] = self.nv_list
        if 'nq_list' not in params_copy:
            params_copy['nq_list'] = self.nq_list
        if 'v_max' not in params_copy:
            params_copy['v_max'] = self.v_max
        if 'q_max' not in params_copy:
            params_copy['q_max'] = self.q_max

        if 'fdm' not in params_copy:
            params_copy['fdm'] = self.fdm
        if 'energy_threshold' not in params_copy:
            params_copy['energy_threshold'] = self.energy_threshold
        if 'energy_bin_width' not in params_copy:
            params_copy['energy_bin_width'] = self.energy_bin_width
        if 'mass_dm' not in params_copy:
            params_copy['mass_dm'] = self.mass_dm
        if 'mass_sm' not in params_copy:
            params_copy['mass_sm'] = self.mass_sm
        
        for idx_bin in range(n_bins):

            if verbose:
                if idx_bin % (n_bins // 5 + 1) == 0:
                    print(
                        f"        Projecting energy bin {idx_bin}/{n_bins-1}...")
            
            energy = (params_copy['energy_threshold'] + 
                      (idx_bin + 0.5) * params_copy['energy_bin_width'])
            params_copy['energy'] = energy
                    
            mcalI = McalI({}, {})
            mcalI.project(params_copy, verbose=False)
            self.mcalIs[idx_bin] = mcalI
        
        self.n_bins = n_bins
        self.l_max = params_copy['l_max']
        self.nv_list = params_copy['nv_list']
        self.nq_list = params_copy['nq_list']
        self.v_max = params_copy['v_max']
        self.q_max = params_copy['q_max']
        self.fdm = params_copy['fdm']
        if verbose:
            end_time = time.time()
            print(f"    Total projection time: {end_time - start_time:.2f} seconds.")


class Rate():
    def __init__(self, vdf, ff, mcalI):
        self.mcalK = self.get_vecK(vdf, ff, mcalI)

    def mu_R(self, wG):
        l_max = min(self.l_max, wG.ellMax)
        lmvmq_max = self.get_lmvmq_index(l_max, l_max, l_max)

        if self.l_mod != wG.lmod:
            raise ValueError("l_mod of Rate and wG do not match.")

        return self.v_max**2 / self.q_max * (
            wG.G_array[:, 0:lmvmq_max+1] @ self.mcalK[0:lmvmq_max+1]
        )

    def get_vecK(self, vdf, ff, mcalI):

        if vdf.v_max != mcalI.v_max:
            raise ValueError("vdf.v_max and mcalI.v_max do not match.")
        self.v_max = vdf.v_max
        if ff.q_max != mcalI.q_max:
            raise ValueError("ff.q_max and mcalI.q_max do not match.")
        self.q_max = ff.q_max

        # get the minimum l_max and output shape
        self.l_max = min(vdf.l_max, ff.l_max, mcalI.l_max)
        self.l_mod = max(vdf.l_mod, ff.l_mod, mcalI.l_mod)
        vecK_shape = (self.get_lmvmq_index(self.l_max, self.l_max, self.l_max)+1,)
        vecK = np.zeros(vecK_shape, dtype=float)

        # get intersection indices for nv and nq
        nv_indices = get_intersection_index(
            vdf_nv=vdf.n_list, mcalI_nv=mcalI.nv_list
        )
        nq_indices = get_intersection_index(
            ff_nq=ff.n_list, mcalI_nq=mcalI.nq_list
        )
        if nv_indices.size == 0 or nq_indices.size == 0:
            return vecK
        
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
            # Shape transformation: (2l+1, Nv) @ (Nv, Nq) @ (Nq, 2l+1) -> (2l+1, 2l+1)
            K_block = self.v_max**3 * V_sub @ I_sub @ F_sub.T

            # --- Step E: Fill back results into vecK ---
            # We need to calculate the corresponding flat indices in vecK
            # Assume vecK is a flattened 1D array
            target_indices = []
            for mv in range(-l, l + 1):
                for mq in range(-l, l + 1):
                    target_indices.append(self.get_lmvmq_index(l, mv, mq))
            
            vecK[target_indices] = K_block.ravel()

        return vecK

    def get_lmvmq_index(self, l, mv, mq):
        return l*(2*l-1)*(2*l+1)//3 + (l+mv)*(2*l+1) + (l+mq)


class BinnedRate():
    def __init__(self, vdf, binnedff, binnedmcalI):
        self.mcalKs = self.get_binned_vecK(vdf, binnedff, binnedmcalI)

    def binned_mu_R(self, wG):
        G_array = np.array(wG.G_array)
        binned_muR = {}
        for idx_bin, vecK in self.mcalKs.items():
            l_max = min(self.l_max, wG.ellMax)
            lmvmq_max = self.get_lmvmq_index(l_max, l_max, l_max)

            if self.l_mod != wG.lmod:
                raise ValueError("l_mod of BinnedRate and wG do not match.")

            binned_muR[idx_bin] = self.v_max**2 / self.q_max * (
                G_array[:, 0:lmvmq_max+1] @ vecK[0:lmvmq_max+1]
            )
        return binned_muR


    def get_binned_vecK(self, vdf, binnedff, binnedmcalI):

        if vdf.v_max != binnedmcalI.v_max:
            raise ValueError("vdf.v_max and mcalI.v_max do not match.")
        self.v_max = vdf.v_max
        if binnedff.q_max != binnedmcalI.q_max:
            raise ValueError("ff.q_max and mcalI.q_max do not match.")
        self.q_max = binnedff.q_max

        self.n_bins = min(binnedff.n_bins, binnedmcalI.n_bins)
        self.l_max = min(vdf.l_max, binnedff.l_max, binnedmcalI.l_max)
        self.l_mod = max(vdf.l_mod, binnedff.l_mod, binnedmcalI.l_mod)

        binned_vecK = {}
        for idx_bin in range(self.n_bins):
            vecK = Rate(
                vdf, 
                binnedff.fnlms[idx_bin], 
                binnedmcalI.mcalIs[idx_bin]
            ).mcalK
            binned_vecK[idx_bin] = vecK
        return binned_vecK

    def get_lmvmq_index(self, l, mv, mq):
        return l*(2*l-1)*(2*l+1)//3 + (l+mv)*(2*l+1) + (l+mq)