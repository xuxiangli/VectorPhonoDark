"""Storage / HDF5 IO layer: the Fnlm and BinnedFnlm coefficient containers."""

import logging

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# Version of the HDF5 layout written by the export_hdf5 methods. Bump when an
# attribute or group layout changes, so future readers can branch on it. Files
# without the attribute predate it and use layout version 1.
HDF5_FORMAT_VERSION = 1


class Fnlm:
    def __init__(
        self,
        l_max: int = -1,
        l_mod: int = 1,
        n_max: int = -1,
        info: dict | None = None,
    ):
        if l_mod not in [1, 2]:
            raise ValueError("l_mod must be either 1 (all l) or 2 (even l only).")

        self.l_max: int = l_max
        self.l_mod: int = l_mod
        self.n_max: int = n_max
        self.info: dict = {} if info is None else info

        self.f_lm_n: np.ndarray = np.array([])

    def get_lm_index(self, ell, m):
        if abs(m) > ell or ell > self.l_max or ell % self.l_mod != 0:
            raise ValueError("Invalid (l, m) values.")
        if self.l_mod == 2:
            return ell * (ell - 1) // 2 + ell + m
        else:
            return ell**2 + ell + m

    def get_n_index(self, n):
        if n > self.n_max:
            raise ValueError("n value exceeds n_max.")
        return n

    def export_hdf5(
        self, filename, groupname, dataname="data", write_info=True, log_info=True
    ):
        if not filename.endswith(".hdf5"):
            filename += ".hdf5"
        with h5py.File(filename, "a") as h5f:
            grp = h5f.require_group(groupname)
            if dataname in grp:
                del grp[dataname]
            dset = grp.create_dataset(dataname, data=self.f_lm_n)
            dset.attrs["format_version"] = HDF5_FORMAT_VERSION
            dset.attrs["l_max"] = self.l_max
            dset.attrs["l_mod"] = self.l_mod
            dset.attrs["n_max"] = self.n_max
            if write_info:
                for key, value in self.info.items():
                    dset.attrs[key] = value
        if log_info:
            logger.info(
                f"Fnlm data written to {filename} in group {groupname}/{dataname}."
            )

    def import_hdf5(
        self, filename: str, groupname: str, dataname: str = "data", log_info=True
    ):
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
                if key not in ["format_version", "l_max", "l_mod", "n_max"]
            }

        if log_info:
            logger.info(
                f"Fnlm data read from {filename} in group {groupname}/{dataname}."
            )

        return self


class BinnedFnlm:
    def __init__(
        self,
        n_bins: int = 0,
        l_max: int = -1,
        l_mod: int = 1,
        n_max: int = -1,
        info: dict | None = None,
    ):
        if l_mod not in [1, 2]:
            raise ValueError("l_mod must be either 1 (all l) or 2 (even l only).")

        self.n_bins: int = n_bins
        self.l_max: int = l_max
        self.l_mod: int = l_mod
        self.n_max: int = n_max
        self.info: dict = {} if info is None else info

        self.fnlms: dict = {}

    def export_hdf5(self, filename, groupname, dataname="data", write_info=True):
        if not filename.endswith(".hdf5"):
            filename += ".hdf5"
        with h5py.File(filename, "a") as h5f:
            grp = h5f.require_group(groupname)
            grp.attrs["format_version"] = HDF5_FORMAT_VERSION
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
                    log_info=False,
                )

        logger.info(
            f"BinnedFnlm data written to {filename} in group {groupname}/bin_*."
        )

    def import_hdf5(self, filename: str, groupname: str, dataname: str = "data"):
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
                if key not in ["format_version", "n_bins", "l_max", "l_mod", "n_max"]
            }
            self.fnlms = {}
            for idx_bin in range(self.n_bins):
                fnlm = Fnlm()
                fnlm.import_hdf5(
                    filename, f"{groupname}/bin_{idx_bin}", dataname, log_info=False
                )
                self.fnlms[idx_bin] = fnlm

        self._check_consistency()

        logger.info(f"BinnedFnlm data read from {filename} in group {groupname}/bin_*.")

        return self

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
